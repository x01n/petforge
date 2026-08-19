"""VITS 本地 TTS 引擎（独立脚本 CLI 协议）。"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

from meapet.log import get_color_logger


log = get_color_logger("tts_vits")

_VITS_LANGUAGE_PREFIX = {
    "zh": "[ZH]",
    "cn": "[ZH]",
    "中文": "[ZH]",
    "zh-cn": "[ZH]",
    "jp": "[JA]",
    "ja": "[JA]",
    "日文": "[JA]",
    "en": "[EN]",
    "英文": "[EN]",
}

# 本地 VITS 音色以日语内置音色为底，但用户要求 TTS 优先中文朗读：
# 未识别语言一律回退中文标签。
_DEFAULT_VITS_PREFIX = "[ZH]"


def _as_text(value: object) -> str:
    return str(value or "").strip()


class TtsVitsMixin:
    """VITS 引擎：vits_infer.py 子进程，按语言包标签合成。"""

    _vits_python: str = ""
    _vits_inprocess: bool = False
    vits_speaker: str = "Mea"
    timeout: float = 60.0

    # 语言上的 VITS 模型支持度：中日英均可合成；默认按中文朗读（用户要求）
    def _vits_language_prefix(self, language: str) -> str:
        return _VITS_LANGUAGE_PREFIX.get(
            _as_text(language).strip().lower(), _DEFAULT_VITS_PREFIX
        )

    @staticmethod
    def _speak_vits(
        host: object,
        text: str,
        output_wav: str,
    ) -> tuple[Optional[str], str]:
        """子进程调用 meapet/tools/vits_infer.py；返回 (wav, 语言标签)。"""
        python_exe = _as_text(getattr(host, "vits_python", ""))
        if not python_exe:
            # 未单独配置 VITS 环境时，复用 GSV 运行时 Python
            python_exe = _as_text(getattr(host, "python_exe", ""))
        script = getattr(host, "vits_infer_script", "")
        if not script:
            script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tools",
                "vits_infer.py",
            )
        language = _as_text(getattr(host, "voice_lang", "") or "")
        prefix = host._vits_language_prefix(language)
        tagged = f"{prefix}{text}"
        command = [
            python_exe,
            script,
            "-t",
            tagged,
            "-o",
            output_wav,
            "--speaker",
            _as_text(getattr(host, "vits_speaker", "Mea")),
        ]
        timeout = getattr(host, "timeout", 60.0)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.error("VITS: 子进程超时")
            return None, ""
        except Exception as exc:
            log.error(f"VITS: 子进程失败 {type(exc).__name__}: {exc}")
            return None, ""
        if result.returncode != 0 or not os.path.isfile(output_wav):
            log.error(f"VITS: 合成失败 {result.stderr[:200]}")
            return None, ""
        return output_wav, _as_text(language)


__all__ = ["TtsVitsMixin"]