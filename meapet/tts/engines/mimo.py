"""MiMo 云端 TTS 引擎（Chat Completions 音频协议）。"""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Optional

from meapet.log import get_color_logger


log = get_color_logger("tts_mimo")

_HEADERS = {
    "Content-Type": "application/json",
    "api-key": "",
}


def _sizable_text(value: object) -> str:
    return str(value or "").strip()


def _project_voice_cache_dir() -> str:
    """项目 voice_cache 目录（兼容测试 patch PROJECT_ROOT 的场景）。"""
    module = __import__("meapet.paths", fromlist=["PROJECT_ROOT"])
    root = getattr(module, "PROJECT_ROOT", None)
    if root is None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[3]
    return os.path.join(os.fspath(root), "voice_cache")


class TtsMimoMixin:
    """MiMo 云端 TTS 引擎：走 Chat Completions 的 audio/response_format 协议。"""

    mimo_api_key: str = ""
    mimo_api_base: str = "https://api.xiaomimimo.com/v1"
    mimo_model: str = "mimo-v2.5-tts"
    mimo_voice: str = "冰糖"
    mimo_style: str = ""
    mimo_clone_ref: str = ""
    mimo_clone_dir: str = ""
    _mimo_voiceclone: bool = False
    _mimo_clone_voice_uri: Optional[str] = None
    _mimo_clone_cache_key: Optional[str] = None
    timeout: float = 60.0

    # ── 固定参考（voice_style 固定 + 按需追加表演） ──

    def _mimo_style_for_mood(
        self,
        mood: str,
        style: str = "",
    ) -> str:
        fixed = _sizable_text(getattr(self, "mimo_style", ""))
        performance = _sizable_text(style)
        if fixed and performance:
            return f"{fixed}\n本句表演：{performance}"
        return fixed or performance

    def _pick_clone_ref_wav(self, _mood: str) -> Optional[str]:
        """依次探明克隆参考音频：显式配置路径 → 项目 voice_cache 目录。"""
        explicit = _sizable_text(
            getattr(self, "mimo_clone_ref", "") or getattr(self, "clone_ref", "")
        )
        if explicit and os.path.isfile(explicit):
            return explicit
        cache_dir = _sizable_text(getattr(self, "mimo_clone_dir", ""))
        if not cache_dir:
            cache_dir = _project_voice_cache_dir()
        if not os.path.isdir(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                return None
        for candidate in (
            os.path.join(cache_dir, "normal_reference.wav"),
            os.path.join(cache_dir, "normal.wav"),
            os.path.join(cache_dir, "reference.wav"),
        ):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return None

    def _mimo_clone_audio_uri(self, ref_wav: str) -> str:
        with open(ref_wav, "rb") as fh:
            raw = fh.read()
        return "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii")

    async def _speak_mimo_async(
        self,
        text: str,
        output_wav: str,
        *,
        mood: str = "neutral",
        lang_tag: str = "",
        style: str = "",
        voice_language: str = "",
    ) -> tuple[Optional[str], str]:
        """异步云端合成；返回 (wav 路径或 None, 语言代码)。"""
        if not _sizable_text(self.mimo_api_key) or not _sizable_text(self.mimo_api_base):
            log.warning("MiMo TTS: 缺少 api_key 或 api_base")
            return None, ""

        ref_wav = None
        if getattr(self, "_mimo_voiceclone", False):
            ref_wav = self._pick_clone_ref_wav(mood)
            if not ref_wav:
                log.warning("MiMo TTS: voice-clone 未配置参考音频")
                return None, ""

        payload: dict = {
            "model": self.mimo_model,
            "messages": [
                {"role": "user", "content": self._mimo_style_for_mood(mood, style) or lang_tag},
                {"role": "assistant", "content": text},
            ],
            "audio": {
                "voice": self.mimo_voice,
                "format": "wav",
            },
            "response_format": "audio",
            "stream": False,
        }
        if ref_wav:
            ref_uri = self._mimo_clone_audio_uri(ref_wav)
            payload["audio"]["voice"] = ref_uri
            payload["audio"]["voice_preset"] = "clone_v2"
            payload["audio"]["cache_key"] = (
                f"{os.path.basename(ref_wav)}-{mood}"
            )

        url = self.mimo_api_base.rstrip("/") + "/chat/completions"
        headers = dict(_HEADERS)
        headers["api-key"] = _sizable_text(self.mimo_api_key)
        import meapet.http_async as http_async

        try:
            response = await http_async.post_json(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except Exception as exc:
            log.error(f"MiMo TTS: 请求失败 {type(exc).__name__}: {exc}")
            return None, ""
        if response.status_code != 200:
            log.error(
                f"MiMo TTS: HTTP {response.status_code} "
                f"{_sizable_text(response.text)[:160]}"
            )
            return None, ""
        try:
            data = response.json()
            audio_b64 = data["choices"][0]["message"]["audio"]["data"]
            wav_bytes = base64.b64decode(audio_b64)
        except Exception as exc:
            log.error(f"MiMo TTS: 响应缺少音频字段 {type(exc).__name__}")
            return None, ""
        os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)
        with open(output_wav, "wb") as fh:
            fh.write(wav_bytes)
        return output_wav, lang_tag

    def _speak_mimo(
        self,
        text,
        output_wav,
        *,
        mood="neutral",
        lang_tag="",
        style="",
        voice_language="",
    ):
        """同步兼容入口；等待异步协程完成。"""
        return asyncio.run(
            self._speak_mimo_async(
                text,
                output_wav,
                mood=mood,
                lang_tag=lang_tag,
                style=style,
                voice_language=voice_language,
            )
        )


__all__ = ["TtsMimoMixin"]