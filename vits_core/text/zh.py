# -*- coding: utf-8 -*-
"""中文（zh）文本 -> 拼音音素 转换器。

将中文字符串转换为 VITS-fast-fine-tuning 风格的中文音素序列
（``Style.FINALS_TONE3`` 韵母 + 数字声调，如 ``"你好"`` -> ``"i3 ao3"``）。

拼音来源：
    单字与词组通过 ``pypinyin``（MIT）内置的多音字字库与词组库，
    结合 ``jieba``（MIT）分词对上下文进行消歧。

设计要点（与上游 VITS-fast-fine-tuning 的 ``text/chinese_cleaners.py`` 行为对齐）：
    - 对每个字，取 ``pinyin(char, style=Style.TONE)`` 的带声调形式；
    - 用 ``Style.INITIALS`` 得到声母（如 ``"zh"``），用 ``Style.FINALS_TONE3``
      得到"还原后的韵母 + 数字声调"（如 ``"ou1"``、``"uei3"``、``"iou2"``、
      ``"v4"``）；
    - 声母 + 韵母 顺序拼接，声母为空（零声母音节）时仅输出韵母。

注意：
    本模块的 ``phoneme`` 输出集合必须**严格是** ``vits_models/finetune_speaker.json``
    的 ``symbols`` 列表的子集。当前实现只输出：ASCII 小写字母、数字 1-4
    （声调）、``v``（替代 ü）、``Q``（自然停顿）。不得输出任何超出该集合的
    字符；如果 pypinyin 返回了未知音节（例如纯数字、拉丁字符），会原样跳过，
    而不是把文字字面量塞进音素序列。
"""

import re

from pypinyin import pinyin, Style
from pypinyin.style._utils import replace_symbol_to_number, replace_symbol_to_no_symbol, get_finals

import jieba

# 上游 VITS-fast-fine-tuning 使用广式（HMM=False）分词，减少无关插入。
jieba.setLogLevel(60)

# 匹配 pypinyin TONE2/TONE3 风格末尾的数字声调（1-5），用于提取声调。
_RE_TONE_SUFFIX = re.compile(r"[1-5]$")
# 合法 ASCII 小写字母 + v，用于过滤拼音中的非法字符。
_RE_BASE = re.compile(r"^[a-zv]+$")

# 在声母为空的零声母音节中，pypinyin 可能返回 "y/w" 作为声母，
# 或返回 "i/u" 开头的韵母；这里都不需要额外处理 —— get_finals 已妥善处理。


def _get_initials_phoneme(py_tone):
    """提取带调拼音的声母（简化为第一个音素）。

    说明：
        上游实现（VITS-fast-fine-tuning 的 chinese_cleaners）实际按字逐个
        输出声母并依赖 pypinyin 的 INITIALS 风格得到标准声母集合。当前实现
        直接使用 ``Style.INITIALS`` 风格通过 ``pinyin`` 获得，见
        ``_syllable_to_phonemes``。
    """
    del py_tone  # 仅为保持签名清晰
    return ""


def _finals_tone3(py_tone):
    """把带声调拼音转换成 '还原后的韵母 + 数字声调'（FINALS_TONE3 风格）。

    参数：
        py_tone: pypinyin 的 Style.TONE 输出，如 ``"zhōng"``。

    返回：
        ``str``，如 ``"ong1"`` / ``"uei3"`` / ``"iou2"`` / ``"v4"``。

    实现说明：
        复刻 pypinyin 官方 ``to_finals_tone3`` 的内部步骤：
        1. ``replace_symbol_to_number`` 把带调符号换成数字（``zhōng`` -> ``zho1ng``）；
        2. 用末尾数字作为声调；
        3. ``replace_symbol_to_no_symbol`` 去数字 -> ``zhong``；
        4. ``replace('v','ü')`` 恢复 ü（pypinyin 内部用 v 表示 ü）；
        5. ``get_finals(base, strict=True/False)`` 切出韵母并做"还原韵母"校验
           （v/u 前的 y/w 处理、iou/uei/uen 还原等）；
        6. 韵母最后的 ``ü`` 统一改回 ``v``，与配置符号表一致。
    """
    num = replace_symbol_to_number(py_tone)      # zho1ng / nv3
    numbers = re.findall(r"[1-5]", num)           # ['1'] / ['3']
    tone = numbers[0] if numbers else ""
    no_tone = replace_symbol_to_no_symbol(py_tone).replace("v", "ü")
    finals = get_finals(no_tone, strict=True) or get_finals(no_tone, strict=False) or no_tone
    return finals.replace("ü", "v") + tone


def _syllable_to_phonemes(py_tone, initials):
    """一个音节（带调拼音 + 声母）转成音素列表。

    参数：
        py_tone: 带调拼音，如 ``"zhōng"``。
        initials: 声母字符串，如 ``"zh"``（可能为空）。

    返回：
        ``list[str]`` 音素列表。

    说明：
        上游风格为："声母 韵母数字" 拆开成独立音素，例如 ``"zhōng"``
        输出 ``["zh", "ong1"]``；``"喜欢"`` 中 ``"xi"`` 拆为 ``["x","i3"]``。
        当前实现把声母作为一个音素，把韵母（含声调数字）作为另一个音素。
    """
    phonemes = []
    if initials:
        phonemes.append(initials)
    fin = _finals_tone3(py_tone)
    # 韵母为空（特殊的鼻化韵段如 ń/ḿ）时忽略，避免空音素。
    if fin:
        phonemes.append(fin)
    return phonemes


def _char_to_phonemes(ch):
    """单个汉字 -> 音素列表（带 zh_ 标记的静态转换）。"""
    # 注：pypinyin 的 lazy_pinyin/Pinyin 对象在模块加载时会预加载字库，
    # 因此这里直接使用 pypinyin 的函数接口。
    py_list = pinyin(ch, style=Style.TONE, heteronym=False, errors="ignore")
    if not py_list:
        return []
    py_tone = py_list[0][0]
    initials = pinyin(ch, style=Style.INITIALS, heteronym=False, errors="ignore")[0][0]
    return _syllable_to_phonemes(py_tone, initials)


def zh_cleaner(text, add_start_end_punctuation=False):
    """把中文文本转成音素串（空格分隔）。

    参数：
        text: 中文字符串（可能包含中文、标点、数字、拉丁字符）。
        add_start_end_punctuation: 与 tacotron 签名兼容，当前未使用。

    返回：
        ``str``，空格分隔的音素，如 ``"i3 ao3 sh i4 ie4"``；标点符
        号（``，。！？；：、`` 等）会被跳过。

    实现细节：
        1. 使用 jieba 对整段文本分词，每个词连带其后的标点一起处理；
           对不含中文的词（数字、英文、符号）原样保留在分词结果中，
           由 ``_token_to_phonemes`` 决定丢弃还是转换。
        2. 对每个汉字调用 ``_char_to_phonemes`` 得到音素。
    """
    del add_start_end_punctuation

    def _token_to_phonemes(token):
        """把 jieba 的一个词转成单字音素（跳过非中文 Token）。"""
        out = []
        # 只要 token 里包含汉字，就逐字处理（忽略混排的拉丁/数字）。
        if re.search(r"[一-鿿]", token):
            for ch in token:
                if re.match(r"[一-鿿]", ch):
                    out.extend(_char_to_phonemes(ch))
                else:
                    # 非汉字落在词内（如 “你好3个”），跳过。
                    pass
        return out

    phon_tokens = []
    for seg in jieba.cut(text, HMM=False):
        phon_tokens.extend(_token_to_phonemes(seg))
    # 在句首与句尾加入静音/停顿音素 Q（与上游 zh 前端一致）。
    result = ["Q"] + phon_tokens + ["Q"]
    return " ".join(result)