# -*- coding: utf-8 -*-
"""日语（ja）文本 -> 音素 转换器。

使用 ``pyopenjtalk``（MIT）的 ``g2p(kana=False)`` 得到"音素级"的发音结果，
如 ``"こんにちは"`` -> ``"k o N n i ch i w a"``。

设计要点（与上游 VITS-fast-fine-tuning 的 ``text/japanese_cleaner.py`` 一致）：
    - ``pyopenjtalk.g2p``（不带 ``kana=True``）返回的是 OpenJTalk 音素表
      中的音素，中间以空格分隔，例如 ``"かきく"`` -> ``"k a k i k u"``；
    - 输出用空格分隔，直接交给 ``_clean_text`` 按空格切分后编号。

符号表约定（当前配置 symbols 包含）：
    - あ行 kana 输出 ``a/i/u/e/o``、``ka/ki/ku/ke/ko`` 等，全部落在
      symbols 的 26 个小写字母集合内；
    - 拨音 ``ん`` -> ``N``、促音 ``っ`` -> ``cl``（symbols 含 ``N``，
      ``cl`` 由 ``c`` + ``l`` 两个符号拼出）；
    - 特殊发音 ``U``（无声化元音）、``Q``（促音停顿）等，若出现在
      symbols 之外，会在 ``cleaners.py`` 中做等价替换。

注意：
    本模块不负责过滤/替换超出 symbols 的音素（那是 ``cleaners.py`` 的职责）。
    它只负责把文本交给 pyopenjtalk，并在输出中保留空格分隔。
"""

import pyopenjtalk


def japanese_cleaner(text):
    """把日语文本转成音素串（空格分隔）。

    参数：
        text: 日语文本（平假名/片假名/汉字混排，pyopenjtalk 内部完成
            词典与读音转换）。

    返回：
        ``str``，空格分隔的音素，如 ``"k o N n i ch i w a"``。

    说明：
        - 使用 ``pyopenjtalk.g2p``（kana=False，即默认），得到音素串；
        - 若文本为空字符串，``g2p`` 返回空串，直接原样返回。
    """
    if not text:
        return ""
    result = pyopenjtalk.g2p(text, kana=False)
    # g2p 偶尔会输出连续多个空格，这里归一化，保证按空格切分时干净。
    return " ".join(result.split())