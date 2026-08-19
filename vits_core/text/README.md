# vits_core/text —— VITS 文本前端（派生实现）

本目录是 `vits_core/` 里缺失的 VITS 文本前端（text frontend）实现，
用于把中/日/英 三种语言的文本转成模型 `vits_models/finetune_speaker.json`
中 `symbols`（68 个）表示的符号编号序列。

## 文件结构

| 文件 | 职责 |
|------|------|
| `symbols.py` | 符号表构建：从配置文件读取 `symbols`，生成 `symbol -> id` 映射 |
| `zh.py` | 中文前端：pypinyin（“声母 + 韵母数字”风格） |
| `ja.py` | 日语前端：pyopenjtalk g2p 音素 |
| `en.py` | 英语前端：eng_to_ipa（带重音的 IPA） |
| `cleaners.py` | 语言标记切分、`_clean_text` 清洗、`text_to_sequence` / `cleaned_text_to_sequence` |
| `__init__.py` | 导出 `text_to_sequence`、`cleaned_text_to_sequence` |

## 语言标记

- `[ZH]...[ZH]` 中文段（pypinyin）
- `[JA]...[JA]` 日语段（pyopenjtalk）
- `[EN]...[EN]` 英语段（eng_to_ipa）
- 未包裹的文本默认按英文处理
- `[KR]`（韩语）在本实现中未展开：若遇到会走英文前端兜底

## 派生来源与许可

本实现是以下开源项目的文本前端在 MIT / Apache-2.0 许可下的派生实现：

- **jaywalnut310/vits**（MIT，Copyright (c) 2021 Jaehyeon Kim）：
  `text/symbols.py`、`text/cleaners.py`（`english_cleaners` 的重音映射
  `ˈ→*`、`ˌ→#`）
- **Plachtaa/VITS-fast-fine-tuning**（Apache-2.0，Copyright (c) 2023 Plachta）：
  `text/symbols.py`、`text/cleaner.py`、`text/cleaners.py`（`cjke_cleaners2`
  的多语种前端逻辑、pypinyin `FINALS_TONE3` 韵母风格、pyopenjtalk 音素）
- **keithito/tacotron**（MIT，Copyright (c) 2017 Keith Ito）：
  文本清洗结构与 IPA 音素映射，`THIRD-PARTY-NOTICE.md` 中已声明
  （`vits_core/text/LICENSE` 为其许可文本位置，本目录不再重复放置）

本目录代码整体以 **MIT License** 分发，与 `THIRD-PARTY-NOTICE.md` 中的
声明保持一致，不新增任何许可证冲突。

## 运行时依赖

实现上述前端时用到的 Python 库：

- `pypinyin`（MIT）：中文拼音与多音字（jieba 分词辅助消歧）
- `pyopenjtalk`（MIT）：日语 g2p（需设置 `OPEN_JTALK_DICT_DIR` 指向
  `dic/open_jtalk_dic_utf_8-1.11`）
- `eng_to_ipa`（MIT）：英语词典转 IPA
- `jieba`（MIT）：中文分词

## 符号集合约束

`symbols` 是模型训练时的嵌入维度来源，本前端输出的每个音素都必须是该
集合的子集。`_clean_text` 里做了统一的等价替换与过滤，确保：

- 日语 `ん→N`、促音 `っ→cl→Q`、无声化 `U→u`（取得 `u`）等
- 英语重音 `ˈ→*`、`ˌ→#`
- 其余字符转小写并过滤成符号表内 ASCII / IPA

如果某个音素确实无法映射到符号表，`text_to_sequence` / 
`cleaned_text_to_sequence` 会抛出 `KeyError`，而不是静默丢弃。