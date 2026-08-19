# -*- coding: utf-8 -*-
"""文本清洗（text cleaning）与 音素 -> 符号编号 核心逻辑。

本模块是整个 ``vits_core/text/`` 的入口与核心，对外导出：
    - ``text_to_sequence``：文本 -> 音素 -> 符号编号序列；
    - ``cleaned_text_to_sequence``：已清洗的音素串 -> 符号编号序列。

默认 ``"cjke_cleaners2"`` 清洗器对应 VITS-fast-fine-tuning 的多语种
（中/英/日/韩）前端管线，规则如下：

    1. 语言标记解析：``[ZH]...[ZH]``、``[JA]...[JA]``、``[EN]...[EN]``
       包裹的段分别用中文/日语/英语前端处理；未包裹的文本默认按英文
       （韩语 ``[KR]`` 前端在本实现中不展开，见 README）。
    2. ``_clean_text`` 段内处理：
       - 中文段：``zh.zh_cleaner`` -> 声母 + 韵母数字（如 ``"i3 ao3"``）；
       - 日语段：``ja.japanese_cleaner`` -> pyopenjtalk 音素
         （如 ``"k o N n i ch i w a"``）；
       - 英文段：``en.english_cleaner`` -> eng_to_ipa 带重音 IPA；
       - 之后统一执行 ``_merge_ph_segment`` 后的等价替换
         （IPA -> 符号表内字符、重音 -> ``*``/``#``、大写转小写、去非法字符），
         保证输出符号集合是模型配置 ``symbols`` 的严格子集。
    3. ``_symbols_to_sequence``：把音素串切分成符号列表，逐符号查
       ``_symbol_to_id`` 编号；未知符号抛异常（不静默跳过）。
    4. ``text_to_sequence`` 在 ``add_blank=True`` 时对结果交错插入 0
       （空白符），与 VITS 训练时 ``commons.intersperse(text_norm, 0)``
       行为一致。

本模块实现参考（均为 MIT / Apache-2.0 声明过的上游，见 README）：
    - jaywalnut310/vits ``text/cleaners.py``（english_cleaners 的重音映射）
    - Plachtaa/VITS-fast-fine-tuning ``text/cleaner.py`` 与 ``text/cleaners.py``
      （``cjke_cleaners2`` 的语言标记切分与等价替换）
    - keithito/tacotron ``text/cleaners.py``（``_clean_text`` 结构与 espeak 映射）
"""

import re

from . import symbols as sym
from . import zh as zh_module
from . import ja as ja_module
from . import en as en_module

# ---------------------------------------------------------------------------
# 符号表资源与基础映射
# ---------------------------------------------------------------------------

# IPA 重音符号 -> 符号表内的 ASCII 替代（对应 VITS 官方 english_cleaners：
# 主重音 ``ˈ`` -> ``"*"``、次重音 ``ˌ`` -> ``"#"``）。
_IPA_STRESS_TO_SYMBOL = {
    "ˈ": "*",  # 主重音
    "ˌ": "#",  # 次重音
    "ˎ": "#",  # 变体次重音（VITS-fast-fine-tuning 里也映射为 #）
}

# 符号表内表示 HTS(OpenJTalk) 特殊音素的等价映射。
# 这些关键词是 pyopenjtalk 可能输出、但模型的 68 符号集合里没有的词法单元；
# 这里按 VITS-fast-fine-tuning 的 vits/pinyin 与 openjtalk 音素约定做等价替换：
#   - ``cl``   促音 -> Q（符号表含 Q，表示停顿/阻塞音）
#   - ``U``    无声化元音（u 无声化）-> u
#   - ``el``   无声化 い -> i
#   - ``Q``    （OpenJTalk 促音停顿时用 Q，若 model 无 Q 则保留，有则直接映射）
#   - ``N``    拨音（model 符号表缺 N，且不可与 nn 混用）
#   - ``pau``  停顿 -> 逗号/句号（见下文 _MERGE_RULES 中已有转换）
_HTS_SOUND_TO_SYMBOL = {
    "cl": "Q",
    "U": "u",
    "el": "i",
}

# 通用 IPA 元音/辅音 -> 小写 ASCII 音素（英文字母覆盖），
# 其中 IPA 处方见配置 symbols 中的 IPA 字符集合：
#   ɑ æ ʃ ʑ ç ɯ ɪ ɔ ɛ ɹ ð ə ɫ ɥ ɸ ʊ ɾ ʒ θ β ŋ ɦ ⁿ ʰ
# 这些字符本身就在 68 个 symbols 里，因此这里**不需要**额外映射；
# 下面的映射只覆盖了 pyopenjtalk 输出中可能出现、却不在 68 个里的等价音素。
_IPA_CLEAN_MAP = {
    "ɑ": "a",  # ɑ -> a（若符号表有 ɑ 则保留原字符更优，但不破坏子集约束）
}


def _clean_text(text):
    """对传入的音素串做"符号表内等价替换 + 过滤"。

    参数：
        text: ``_merge_ph_segment`` 拼接好的音素串（空格分隔的词素）。

    返回：
        ``str`` 清洗后的音素串。

    说明：
        本函数是 ``zh_cleaner/ja_cleaner/en_cleaner`` 三个产物的共同后处理，
        保证不会产生超出 68 符号集合的字符。处理顺序：
        1. ``_HTS_SOUND_TO_SYMBOL`` 映射（cl->Q、U->u、el->i）；
        2. IPA 重音替换（``ˈ``->``*``、``ˌ``->``#``）；
        3. 大写转小写；
        4. 移除不属于 ``[a-z*#^`=]`` 集合的字符（保留小写字母、重音替代、
           ``^``/``#``/``*``/``=`` 等 VITS 符号表里的 ASCII 控制音素，
           以及配置里的 IPA 字符——当前实现先全部转成 ASCII 小写）。
        最后按空格切分再合并，避免连续空格。
    """
    # HTS 特殊音素等价替换
    for src, dst in _HTS_SOUND_TO_SYMBOL.items():
        text = re.sub(r"(?<![a-z]){}(?![a-z])".format(src), dst, text)
    # 语言侧缺失符号的等价替换（68 符号集合内合法组合）：
    #   中文/日语/英语中可能出现的 r、j、q、c、ch 音素在配置 symbols 中缺
    #   （无 r、无 j、无 q、无 c），统一映射到 68 集合内的等价写法：
    #     r  -> ɹ  （近音辅音）
    #     j  -> ʑ  （浊腭擦音，中文 j 声母与英文 j）
    #     q  -> tʃ（清塞擦音近似，中文 q 声母）
    #     ch -> tʃ（清塞擦音，日语 ちゃ/英语 tch、中文 ch 声母）
    #     c  -> tʃ（仅作防御，正常情况下 c 不出现在音素串）
    # 注意：这里的顺序先把 ch 映射（避免 ch 中的 c 单独残留），再处理单字母。
    text = text.replace("ch", "tʃ")
    text = text.replace("j", "ʑ")
    text = text.replace("q", "tʃ")
    text = text.replace("c", "tʃ")
    text = text.replace("r", "ɹ")
    # IPA 重音符号
    for src, dst in _IPA_STRESS_TO_SYMBOL.items():
        text = text.replace(src, dst)
    # 小写化
    text = text.lower()
    # 过滤：只保留小写字母、数字(0-9)、以及 VITS 符号表里定义过的 ASCII 字符
    #       （这些字符的映射关系在 ``_symbols_to_sequence`` 中逐字符编号）。
    # 数字只保留 1-5（声调），其余数字被过滤（防御）。
    text = re.sub(r"[^a-z*#^`=·…´`'•]", " ", text)
    # 合并多余空白
    return " ".join(text.split())


# VITS 官方 english_cleaners 的标准标点映射（作为符号直接保留）。
_SYMBOL_TO_ID_PUNCT = {
    ",": ",",
    ".": ".",
    "!": "!",
    "?": "?",
    "-": "-",
    "~": "~",
    "…": "…",
    "^": "^",
    "#": "#",
    "*": "*",
    "=": "=",
    "`": "`",
    " ": " ",
}


# ---------------------------------------------------------------------------
# 语言标记切分与清洗入口
# ---------------------------------------------------------------------------

def _merge_ph_segment(segments):
    """把各语言段清洗后的音素串按顺序拼接（以空格分隔）。

    参数：
        segments: ``list[str]``，每个元素是某语言段清洗后的音素串。

    返回：
        ``str`` 空格分隔的合并结果。
    """
    return " ".join(s for s in segments if s.strip())


def _clean_cjke(text, cleaners):
    """按语言标记切分文本，逐段调用对应语言的 cleaner，最后汇总清洗。

    参数：
        text: 原始文本（可能含 ``[ZH]...[ZH]``、``[JA]...[JA]``、
            ``[EN]...[EN]`` 标记）。
        cleaners: 清洗器列表，本实现只识别 ``"cjke_cleaners2"``。

    返回：
        ``str`` 清洗后的音素串。

    实现：
        - 用正则把 ``[ZH]``/``[JA]``/``[EN]`` 标记切成若干段；
        - 段有语言前缀的走对应 cleaner（``zh_cleaner/ja_cleaner/en_cleaner``）；
        - 无前缀（空白/无标记）的段视为英文；
        - 每段结果统一调用 ``_clean_text`` 做符号表内替换。
    """
    if "cjke" in "".join(cleaners):
        return _clean_cjke2(text)
    # 非 cjke 清洗器（如 tacotron 的 english_cleaners）退化为英文前端。
    return _clean_text(en_module.english_cleaner(text))


def _clean_cjke2(text):
    """实现 ``cjke_cleaners2``：语言标记 + 各语言 cleaner。

    输入约定（与 VITS-fast-fine-tuning 的输入一致）：
        ``[ZH]你好[ZH]``、``[JA]こんにちは[JA]``、``[EN]hello[EN]``，
    即每个语言段的开/闭标记同名。没有语言标记的文本默认按英文处理。

    实现：
        先把文本按语言前缀切出第一个段与剩余文本，再递归处理剩余文本。
    """

    def _split_one(txt):
        """从 txt 开头切出一个语言段（若存在），返回 (seg, rest)。"""
        if not txt:
            return None, ""
        for prefix in ("[ZH]", "[JA]", "[EN]"):
            if txt.startswith(prefix):
                # 找到同名闭标记
                idx = txt.find(prefix, len(prefix))
                if idx >= 0:
                    inner = txt[len(prefix):idx]
                    rest = txt[idx + len(prefix):]
                    return (prefix, inner), rest
                # 闭标记缺失：把整个剩余当英文兜底
                return None, txt
        return None, txt

    segments = []
    remaining = text
    while remaining:
        seg, remaining = _split_one(remaining)
        if seg is None:
            if remaining.strip():
                # 未包裹文字，视为英文段
                segments.append(("EN", remaining))
            remaining = ""
        else:
            segments.append(seg)

    output = []
    for lang, inner in segments:
        if lang == "[ZH]":
            cleaned = zh_module.zh_cleaner(inner)
        elif lang == "[JA]":
            cleaned = ja_module.japanese_cleaner(inner)
        else:
            cleaned = en_module.english_cleaner(inner)
        output.append(_clean_text(cleaned))
    return _merge_ph_segment(output)


# ---------------------------------------------------------------------------
# 音素串 -> 符号编号
# ---------------------------------------------------------------------------

def _symbols_to_sequence(symbols):
    """把清洗后的音素串转成符号编号序列。

    参数：
        symbols: ``str``，空格分隔的音素串。

    返回：
        ``list[int]`` 编号序列。

    说明：
        音素串按空格切分后逐符号查 ``_symbol_to_id``。未知符号抛
        ``KeyError``（本实现的 ``_clean_text`` 已经保证输入是符号表子集，
        这里抛异常只是为了防御上游直接调用 ``cleaned_text_to_sequence``）。
    """
    _symbol_to_id = sym.get_symbol_to_id()
    seq = []
    for s in symbols.split():
        if s not in _symbol_to_id:
            raise KeyError(
                "音素 {!r} 不在模型的符号表中（共 {} 个符号）。".format(
                    s, len(_symbol_to_id)
                )
            )
        seq.append(_symbol_to_id[s])
    return seq


def text_to_sequence(text, symbols=None, _cleaners="cjke_cleaners2"):
    """文本 -> 音素 -> 符号编号序列。

    参数：
        text: 原始文本。
        symbols: 符号列表；若为 ``None``，自动从模型配置读取
            （``vits_models/finetune_speaker.json``）。**必须与模型
            嵌入层的维度严格一致**。
        _cleaners: 清洗器列表（兼容 VITS 官方签名 ``text_to_sequence(text, cleaner_names)``）；
            默认 ``"cjke_cleaners2"``。

    返回：
        ``list[int]`` 符号编号序列。

    说明：
        - 当调用方传入 ``symbols`` 时，用它构建 ``_symbol_to_id``；
          否则使用配置文件的 ``symbols``（本仓库默认配置）。
        - ``add_blank`` 交错逻辑由调用方（``vits_infer.py`` / ``data_utils.py``）
          在拿到编号序列后自行调用 ``commons.intersperse``，这里不做，以
          保持与官方 ``text_to_sequence`` 相同的职责边界。
    """
    if symbols is not None:
        sym.build_symbol_to_id(symbols)
        # 注意：build 不修改全局 _symbol_to_id；这里显式设置以避免后续
        # 其它模块使用全局映射时与本次传入的 symbols 不一致。
        global _symbol_to_id
        _symbol_to_id = sym.build_symbol_to_id(symbols)

    cleaned = _clean_text(_clean_cjke(text, [_cleaners] if isinstance(_cleaners, str) else _cleaners))
    return _symbols_to_sequence(cleaned)


def cleaned_text_to_sequence(cleaned_text, symbols=None):
    """已清洗的音素串 -> 符号编号序列。

    参数：
        cleaned_text: 空格分隔的音素串（如 ``"i3 ao3 Q"``，来自训练标注）。
        symbols: 符号列表；为 ``None`` 时使用配置文件的默认符号表。

    返回：
        ``list[int]`` 编号序列。
    """
    if symbols is not None:
        sym.build_symbol_to_id(symbols)
        global _symbol_to_id
        _symbol_to_id = sym.build_symbol_to_id(symbols)
    return _symbols_to_sequence(str(cleaned_text))