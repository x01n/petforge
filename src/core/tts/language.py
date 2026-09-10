"""TTS 语言标签与参考音频命名约定。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

# 这些标签与项目交付的参考音频文件名一致：``zh_*``、``jp_*`` 和
# ``en_*``。其它语言不会被静默映射到已有音色。
_LANGUAGE_ALIASES = {
    "zh": "zh",
    "cn": "zh",
    "zh-cn": "zh",
    "zh_cn": "zh",
    "chinese": "zh",
    "中文": "zh",
    "汉语": "zh",
    "ja": "jp",
    "jp": "jp",
    "jpn": "jp",
    "ja-jp": "jp",
    "ja_jp": "jp",
    "japanese": "jp",
    "日文": "jp",
    "日语": "jp",
    "en": "en",
    "eng": "en",
    "en-us": "en",
    "en_us": "en",
    "english": "en",
    "英文": "en",
    "英语": "en",
}

_PROTOCOL_LANGUAGE_DEFAULTS = {
    "zh": "zh",
    # 交付参考音频使用 jp 前缀；GPT-SoVITS api_v2 的语言枚举使用 ja。
    "jp": "ja",
    "en": "en",
}
# 允许中文情绪标签（例如“开心”），但明确排除点号、斜杠和空白，避免标签
# 被拼接为资源路径时越界。语言协议是否支持由 canonical 映射和服务端校验负责。
_SAFE_TOKEN_PATTERN = re.compile(r"^[\w]+(?:-[\w]+)*$", re.UNICODE)


def is_safe_tts_token(value: object) -> bool:
    """判断语言/情绪标签是否能安全用于协议字段和资源文件名。"""

    raw = str(value or "").strip().lower().replace("_", "-")
    return bool(raw and _SAFE_TOKEN_PATTERN.fullmatch(raw))


def canonical_tts_language(value: object) -> str:
    """规范化项目支持的语言桶；未知标签只保留主标签。"""

    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    normalized = raw.replace("_", "-")
    direct = _LANGUAGE_ALIASES.get(raw) or _LANGUAGE_ALIASES.get(normalized)
    if direct:
        return direct
    if not is_safe_tts_token(raw):
        return ""
    return normalized.split("-", 1)[0]


def reference_language_prefix(value: object) -> str:
    """返回参考音频文件名使用的语言前缀。"""

    return canonical_tts_language(value)


def protocol_tts_language(value: object) -> str:
    """返回 GPT-SoVITS 请求使用的语言标签。

    资源文件名和服务协议的标签不是同一个命名空间；未知标签保留调用方
    的规范化主标签，具体部署若使用其它枚举可通过配置字段覆盖。
    """

    raw = str(value or "").strip()
    canonical = canonical_tts_language(raw)
    return _PROTOCOL_LANGUAGE_DEFAULTS.get(canonical, canonical)


def normalize_reference_audios(value: object) -> dict[str, dict[str, str]]:
    """规范化按语言配置的参考音频映射。

    每项支持字符串路径，或包含 ``path``/``text``/``prompt_lang``/``text_lang`` 的映射。
    同一语言通过别名重复配置时直接报错，避免选择结果依赖 YAML 顺序。
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("tts.reference_audios must be a mapping")
    result: dict[str, dict[str, str]] = {}
    for raw_language, raw_entry in value.items():
        if not isinstance(raw_language, str):
            raise ValueError("tts.reference_audios language keys must be strings")
        language = canonical_tts_language(raw_language)
        if not language:
            raise ValueError("tts.reference_audios contains an empty language")
        if language in result:
            raise ValueError(f"duplicate TTS reference language: {language}")
        if isinstance(raw_entry, Mapping):
            raw_path = raw_entry.get("path", raw_entry.get("ref_audio_path", ""))
            raw_text = raw_entry.get("text", raw_entry.get("prompt_text", ""))
            raw_prompt_lang = raw_entry.get("prompt_lang", "")
            raw_text_lang = raw_entry.get("text_lang", "")
            for field_name, field_value in (
                ("path", raw_path),
                ("text", raw_text),
                ("prompt_lang", raw_prompt_lang),
                ("text_lang", raw_text_lang),
            ):
                if field_value is not None and not isinstance(field_value, str):
                    raise ValueError(
                        f"tts.reference_audios.{language}.{field_name} must be a string"
                    )
            path = str(raw_path or "").strip()
            text = str(raw_text or "").strip()
            prompt_lang = str(raw_prompt_lang or "").strip()
            text_lang = str(raw_text_lang or "").strip()
        else:
            if not isinstance(raw_entry, str):
                raise ValueError(f"tts.reference_audios.{language} must be a string or mapping")
            path = raw_entry.strip()
            text = ""
            prompt_lang = ""
            text_lang = ""
        result[language] = {
            "path": path,
            "text": text,
            "prompt_lang": prompt_lang,
            "text_lang": text_lang,
        }
    return result


def resolve_mood_reference(
    ref_dir: str | Path,
    *,
    language: object,
    mood: object,
) -> dict[str, str] | None:
    """解析 ``<mood>/<lang>_<mood>.wav`` 及同名文本。

    情绪目录缺失时仅回退到 ``normal``，语言前缀始终保持请求语言；不会
    把日语请求回退到中文或其它语言音频。
    """

    raw_root = str(ref_dir or "").strip()
    if not raw_root:
        return None
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        return None
    prefix = reference_language_prefix(language)
    if not prefix:
        return None
    requested_mood = str(mood or "normal").strip().lower() or "normal"
    if not is_safe_tts_token(requested_mood):
        return None
    moods = (requested_mood,) if requested_mood == "normal" else (requested_mood, "normal")
    for mood_name in moods:
        folder = root / mood_name
        wav_path = (folder / f"{prefix}_{mood_name}.wav").resolve()
        if not wav_path.is_relative_to(root):
            continue
        if not wav_path.is_file():
            continue
        text_path = wav_path.with_suffix(".txt").resolve()
        text = ""
        if text_path.is_relative_to(root) and text_path.is_file():
            try:
                text = text_path.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
        return {
            "path": str(wav_path),
            "text": text,
            "prompt_lang": protocol_tts_language(prefix),
        }
    return None


__all__ = [
    "canonical_tts_language",
    "is_safe_tts_token",
    "normalize_reference_audios",
    "reference_language_prefix",
    "protocol_tts_language",
    "resolve_mood_reference",
]
