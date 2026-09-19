from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pytest

from services.asr import ASRService

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ASR_PYTHON = _PROJECT_ROOT / "temp/third_party/GPT-SoVITS/.venv/bin/python"
_MODEL_PATH = Path(
    os.environ.get(
        "MEAPET_ASR_MODEL_DIR",
        "/home/clfchen/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master",
    )
)


@pytest.mark.skipif(
    os.environ.get("MEAPET_ASR_MODEL_DIR") is not None,
    reason="MEAPET_ASR_MODEL_DIR 由交付安装覆盖；本地默认位置测试仅适用默认环境",
)
def test_local_model_path_matches_bundled_default() -> None:
    """默认未覆盖时 _MODEL_PATH 必须保持本机交付的默认 modelscope 位置。"""

    assert str(_MODEL_PATH) == (
        "/home/clfchen/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master"
    )


_REFERENCE_WAV = _PROJECT_ROOT / "resources/GPT-Sovits/soft/zh_soft.wav"
_REFERENCE_TEXT = _PROJECT_ROOT / "resources/GPT-Sovits/soft/zh_soft.txt"


@pytest.mark.skipif(
    not (
        _ASR_PYTHON.is_file()
        and _MODEL_PATH.is_dir()
        and all(
            (_MODEL_PATH / filename).is_file()
            for filename in ("model.pt", "config.yaml", "tokens.json", "am.mvn")
        )
        and _REFERENCE_WAV.is_file()
        and _REFERENCE_TEXT.is_file()
    ),
    reason="local SenseVoice runtime or reference audio is unavailable",
)
def test_local_sensevoice_worker_transcribes_reference_wav_without_text_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        environment = {
            key: os.environ[key]
            for key in (
                "HOME",
                "PATH",
                "LANG",
                "LC_ALL",
                "LC_CTYPE",
                "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "TMPDIR",
                "XDG_CACHE_HOME",
            )
            if key in os.environ
        }
        source_root = str(_PROJECT_ROOT / "src")
        environment.update(
            {
                "PYTHONPATH": source_root,
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        service = ASRService(
            (
                str(_ASR_PYTHON),
                "-u",
                "-m",
                "services.asr.worker",
                "--backend",
                "sensevoice",
                "--model-path",
                str(_MODEL_PATH),
                "--device",
                "cpu",
                "--language",
                "auto",
                "--max-audio-bytes",
                str(4 * 1024 * 1024),
            ),
            enabled=True,
            backend="sensevoice",
            model_name="SenseVoiceSmall",
            device="cpu",
            language="auto",
            timeout_seconds=60.0,
            startup_timeout_seconds=180.0,
            max_audio_bytes=4 * 1024 * 1024,
            cwd=_PROJECT_ROOT,
            env=environment,
        )
        expected = _REFERENCE_TEXT.read_text(encoding="utf-8").strip()
        try:
            health = await service.start()
            assert health.ready is True
            result = await service.transcribe(_REFERENCE_WAV.read_bytes(), language="zh")
            assert expected
            assert result.text
            assert any("\u4e00" <= char <= "\u9fff" for char in result.text)
            assert result.language == "zh"
            assert result.confidence is None
            assert result.confidence_available is False
            assert expected not in caplog.text
            assert result.text not in caplog.text
        finally:
            await service.aclose()

    with caplog.at_level(logging.DEBUG):
        asyncio.run(scenario())
