# -*- coding: utf-8 -*-
"""符号表（symbols）与 符号->编号（symbol_to_id）映射。

本模块只提供符号表的**构建逻辑**，不内置具体的符号列表。
符号列表的唯一来源是模型配置文件 ``vits_models/finetune_speaker.json`` 中的
``symbols`` 字段：训练时模型嵌入层的维度（``len(symbols)``）必须与实际
提供的符号列表严格一致，任何多一个/少一个符号都会导致推理时的维度错误。

本实现参考（均为 MIT / Keith Ito tacotron 文本前端，且已在
THIRD-PARTY-NOTICE.md 中声明）：
- jaywalnut310/vits 的 ``text/symbols.py``（``symbols`` 常量 + ``symbol_to_id``）
- keithito/tacotron 的 ``util.py`` / ``symbols.py``（``symbols`` 的组成约定：
  ``_punctuation`` + 字母表 + 国际音标（IPA）集合）
- Plachtaa/VITS-fast-fine-tuning 的 ``text/symbols.py``（多语种 CJKE 符号表来源）

依赖库：无。

本模块不允许对符号做任何形式的猜测或补集：如果配置给出的符号出现缺失，
必须显式抛出异常，而不是静默跳过或者把文字字面量塞进序列。
"""

import json
import os

# 已经从 JSON 配置文件构建好的"符号 -> 编号"映射（全局缓存，避免重复读盘）。
_symbol_to_id: dict = {}


def build_symbol_to_id(symbols):
    """根据给定的符号列表构建 ``{symbol: id}`` 映射。

    参数：
        symbols: 可迭代对象，元素为字符串，通常来自模型配置里的 ``symbols`` 列表。

    返回：
        ``dict``，键为符号字符串，值为其下标（0 起的整数）。

    说明：
        - 下标从 0 开始，与 ``text_to_sequence`` 中 "0 表示空白符（blank）" 的
          约定保持一致（VITS 训练时 ``add_blank`` 交错插入的空白符 id 就是 0）。
        - 同一个符号出现在多个位置时，以第一次出现的下标为准（防御性处理，
          正常配置中符号不重复）。
    """
    mapping = {}
    for idx, symbol in enumerate(symbols):
        if symbol not in mapping:
            mapping[symbol] = idx
    return mapping


def _load_symbols_from_config():
    """从模型配置目录读取 ``finetune_speaker.json`` 的 ``symbols`` 字段。

    返回：
        ``list[str]`` 符号列表。

    说明：
        - 配置路径按 VITS-fast-fine-tuning 的约定固定为
          ``{$项目根}/vits_models/finetune_speaker.json``。
        - 通过 ``__file__`` 定位项目根，避免依赖当前工作目录。
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(root, "vits_models", "finetune_speaker.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            "找不到模型配置文件: {}（文本前端依赖其 symbols 字段）".format(config_path)
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    symbols = config.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError(
            "模型配置文件 {} 中缺少非空的 symbols 列表".format(config_path)
        )
    return symbols


def _ensure_loaded():
    """确保全局 ``_symbol_to_id`` 已按配置文件构建。"""
    global _symbol_to_id
    if not _symbol_to_id:
        _symbol_to_id = build_symbol_to_id(_load_symbols_from_config())
    return _symbol_to_id


def get_symbol_to_id():
    """返回全局的 ``{symbol: id}`` 映射（首次调用时自动从配置加载）。"""
    return _ensure_loaded()


def get_symbols():
    """返回全局的符号列表（按配置中的顺序）。"""
    return list(_ensure_loaded().keys())


def symbol_to_id(symbol):
    """单个符号 -> 编号。未知符号会抛 ``KeyError``。

    参数：
        symbol: 符号字符串。

    返回：
        ``int`` 编号。

    异常：
        ``KeyError``: 符号不在配置符号表中。
    """
    mapping = _ensure_loaded()
    if symbol not in mapping:
        raise KeyError(
            "符号 {!r} 不在模型的符号表中（共 {} 个符号）。"
            "这可能是因为文本前端产生了一个超出符号表集合的音素。".format(
                symbol, len(mapping)
            )
        )
    return mapping[symbol]