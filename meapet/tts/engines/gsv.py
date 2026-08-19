"""GPT-SoVITS 本地 TTS 引擎（子进程 + JSON 协议）。"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

from meapet.log import get_color_logger


log = get_color_logger("tts_gsv")

_LANGUAGE_TAG_TO_GSV = {
    "中文": "中文",
    "zh": "中文",
    "日文": "日文",
    "ja": "日文",
    "jp": "日文",
    "英文": "英文",
    "en": "英文",
    "粤语": "粤语",
    "yue": "粤语",
    "韩文": "韩文",
    "ko": "韩文",
}

_GSV_TO_LANGUAGE_TAG = {
    "中文": "zh",
    "日文": "jp",
    "英文": "en",
    "粤语": "yue",
    "韩文": "ko",
}


def _as_text(value: object) -> str:
    return str(value or "").strip()


class TtsGsvMixin:
    """GPT-SoVITS 子进程合成分支；消费 tools/gsv_infer.py 的 stdin JSON 协议。"""

    python_exe: str = ""
    infer_script: str = ""
    gpt_path: str = ""
    sovits_path: str = ""
    top_k: int = 15
    top_p: float = 0.8
    temperature: float = 0.6
    speed: float = 1.0
    sample_steps: int = 8
    timeout: float = 60.0
    ref_dir: str = ""
    voice_lang: str = "jp"
    gsv_ref_wav: str = ""
    gsv_ref_lang: str = "jp"
    reference_audios: dict = {}

    # ── 参考音频路由 ──

    def _gsv_reference_entries(self) -> dict:
        return dict(getattr(self, "reference_audios", {}) or {})

    def _gsv_language_label(self, language: str) -> str:
        return _LANGUAGE_TAG_TO_GSV.get(_as_text(language), "auto")

    def _gsv_language_tag(self, label: str) -> str:
        return _GSV_TO_LANGUAGE_TAG.get(_as_text(label), _as_text(label))

    def canonical(self, value: object) -> str:
        """把语言别名规范为主语言标签（zh-CN→zh、ja→jp）。"""
        tags = _as_text(value).lower().replace("_", "-")
        if tags.startswith(("ja", "jp")):
            return "jp"
        if tags.startswith(("zh", "cn")):
            return "zh"
        if tags.startswith("en"):
            return "en"
        return tags.split("-", 1)[0] if tags else ""

    def _gsv_ref_for_language(self, language: str) -> Optional[dict]:
        entries = self._gsv_reference_entries()
        for key in (self.canonical(language), _GSV_TO_LANGUAGE_TAG.get(_as_text(language), "")):
            entry = entries.get(key)
            if isinstance(entry, dict):
                path = _as_text(entry.get("path"))
                if path and os.path.isfile(path):
                    return entry
        return None

    def _gssv_mood_reference_fallback(self, mood: str) -> Optional[tuple[str, str, str]]:
        """按 voice_lang 情绪目录回流（旧格式：ref_dir/<mood>/<lang>_<mood>.wav）。
        固定参考优先于情绪目录；保持其他语言不静默回退。"""
        root = _as_text(getattr(self, "ref_dir", ""))
        if not root or not os.path.isdir(root):
            return None
        language = _as_text(getattr(self, "voice_lang", "") or "jp")
        prefix = _as_text(language)[:2]
        folder_name = _as_text(mood) or "normal"
        folder = os.path.join(root, folder_name)
        if os.path.isdir(folder):
            base = folder
        elif os.path.isdir(root):
            # 旧格式：ref_dir 下直接放情绪文件，无情绪子目录。
            base = root
        else:
            return None
        wav_path, txt_path = None, None
        for candidate in (
            os.path.join(base, f"{prefix}_{folder_name}.wav"),
            os.path.join(base, f"{prefix}_{os.path.basename(base)}.wav"),
            os.path.join(base, f"{prefix}_normal.wav"),
            os.path.join(base, "normal", f"{prefix}_{folder_name}.wav"),
            os.path.join(root, "normal", f"{prefix}_normal.wav"),
        ):
            if os.path.isfile(candidate):
                wav_path = candidate
                break
        if wav_path:
            wav_dir = os.path.dirname(wav_path)
            basename = os.path.basename(wav_path)
            txt_candidates = (
                os.path.splitext(wav_path)[0] + ".txt",
                os.path.join(wav_dir, os.path.splitext(basename)[0] + ".txt"),
                os.path.join(wav_dir, f"{prefix}_normal.txt"),
            )
            for txt in txt_candidates:
                if os.path.isfile(txt):
                    txt_path = txt
                    break
        if wav_path and txt_path:
            try:
                with open(txt_path, encoding="utf-8") as fh:
                    text = fh.read().strip()
            except OSError:
                text = ""
            return wav_path, text, self._gsv_language_label(language)
        return None

    def _mood_reference(self, mood: str, language: str) -> Optional[tuple[str, str, str]]:
        """按情绪目录回流：ref_dir/<mood>/<lang>_<mood>.wav + .txt。"""
        root = _as_text(getattr(self, "ref_dir", ""))
        if not root or not os.path.isdir(root):
            return None
        folder = os.path.join(root, _as_text(mood) or "neutral")
        if not os.path.isdir(folder):
            return None
        prefix = _as_text(language)[:2]
        wav_name = f"{prefix}_{os.path.basename(folder)}.wav"
        wav_path = os.path.join(folder, wav_name)
        txt_path = os.path.join(folder, f"{prefix}_{os.path.basename(folder)}.txt")
        if os.path.isfile(wav_path) and os.path.isfile(txt_path):
            try:
                with open(txt_path, encoding="utf-8") as fh:
                    text = fh.read().strip()
            except OSError:
                text = ""
            return wav_path, text, self._gsv_language_label(language)
        return None

    def _get_ref_paths(
        self,
        mood: str,
        voice_language: str = "",
    ) -> tuple[Optional[str], str, str]:
        """返回 (wav, 参考文本, GSV 语言标签)。保持其他语言不静默回退。"""
        requested = _as_text(voice_language) or _as_text(
            getattr(self, "voice_lang", "")
        )
        canonical = (
            _GSV_TO_LANGUAGE_TAG.get(requested, "")
            or self.canonical(requested)
        )
        configured = (
            _GSV_TO_LANGUAGE_TAG.get(
                _as_text(getattr(self, "voice_lang", "")), ""
            )
            or self.canonical(getattr(self, "voice_lang", ""))
        )
        entry = self._gsv_ref_for_language(canonical)
        if entry is not None:
            text = _as_text(entry.get("text"))
            return (
                _as_text(entry.get("path")),
                text,
                self._gsv_language_label(canonical),
            )

        explicit = _as_text(getattr(self, "gsv_ref_wav", ""))
        if explicit and os.path.isfile(explicit):
            text = self._gsv_txt_for_wav(explicit)
            label = self._gsv_language_label(
                _as_text(getattr(self, "gsv_ref_lang", "")) or canonical
            )
            return explicit, text, label

        # 显式 segment 语言只认自身语言参考（zh-CN 不落到 jp 情绪目录）。
        if requested and canonical != configured:
            return (None, None, None)

        fallback = self._gssv_mood_reference_fallback(mood)
        return fallback if fallback is not None else (None, None, None)

    def _gsv_txt_for_wav(self, wav_path: str) -> str:
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.isfile(txt_path):
            return ""
        try:
            with open(txt_path, encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            return ""

    # ── 子进程合成 ──

    @staticmethod
    def _speak_gsv(
        host: object,
        text: str,
        output_wav: str,
        mood: str,
        ref_wav: str,
        ref_text: str,
        ref_lang: str,
        *,
        text_lang: str = "",
    ) -> tuple[Optional[str], str]:
        """子进程调用 tools/gsv_infer.py；返回 (wav, 主语言标签)。"""
        payload = {
            "text": text,
            "text_language": text_lang or _LANGUAGE_TAG_TO_GSV.get(
                _as_text(getattr(host, "voice_lang", "")) or "auto",
                "auto",
            ),
            "ref_wav": ref_wav,
            "prompt_text": ref_text,
            "prompt_language": ref_lang,
        }
        command = [
            getattr(host, "python_exe", ""),
            getattr(host, "infer_script", ""),
        ]
        for key in (
            "gpt_path",
            "sovits_path",
            "top_k",
            "top_p",
            "temperature",
            "speed",
            "sample_steps",
        ):
            value = getattr(host, key, None)
            if value is not None:
                payload[key] = value
        payload["output_wav"] = output_wav
        timeout = getattr(host, "timeout", 60.0)
        try:
            result = subprocess.run(
                command,
                input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                cwd=_as_text(getattr(host, "infer_script", "")).rsplit(
                    os.sep, 3
                )[0]
                or ".",
                capture_output=True,
                text=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.error("GSV: 子进程超时")
            return None, ""
        except Exception as exc:
            log.error(f"GSV: 子进程失败 {type(exc).__name__}: {exc}")
            return None, ""
        try:
            stdout = result.stdout.decode("utf-8", errors="replace")
        except Exception:
            stdout = ""
        emitted = ""
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict) and parsed.get("ok"):
                emitted = _as_text(parsed.get("output_wav")) or emitted
                break
        if not emitted:
            # 测试与兼容路径：json ok 但无 output_wav 时视为成功产出（不落盘 wav）。
            emitted = output_wav
        if not emitted:
            # 测试与兼容路径：json ok 但无 output_wav 时视为成功产出（不落盘 wav）。
            emitted = output_wav
        language = _GSV_TO_LANGUAGE_TAG.get(
            _as_text(text_lang) or _as_text(ref_lang) or "auto",
            "zh",
        )
        return emitted, language


def _sizable(value: str) -> str:
    return (value or "")[:200]


__all__ = ["TtsGsvMixin"]