# -*- coding: utf-8 -*-
"""vits_core/text —— VITS 文本前端（text frontend）包。

将输入文本转换为模型可用的符号编号序列：
    - 中文：pypinyin（声母 + 韵母数字，如 ``"你好"`` -> ``"i3 ao3"``）
    - 日语：pyopenjtalk（如 ``"こんにちは"`` -> ``"k o N n i ch i w a"``）
    - 英语：eng_to_ipa（带重音的 IPA，重音映射为 ``*``/``#``）

对外导出（与 VITS 官方 ``text/__init__.py`` 一致）：
    - ``text_to_sequence(text, symbols, cleaners)``
    - ``cleaned_text_to_sequence(cleaned_text, symbols)``

符号表唯一来源为模型配置 ``vits_models/finetune_speaker.json`` 的
``symbols`` 字段（由 ``symbols.py`` 自动加载）。
"""

from .cleaners import text_to_sequence, cleaned_text_to_sequence

__all__ = ["text_to_sequence", "cleaned_text_to_sequence"]