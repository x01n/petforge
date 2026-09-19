"""基于实际文件的运行资源能力探测。"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path

from core.rendering.resources import is_valid_webp, validate_model3_description

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioAsset:
    path: Path
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class ResourceInventory:
    root: Path
    live2d_models: tuple[Path, ...]
    sprite_count: int
    gsv_weight_files: tuple[Path, ...]
    reference_voices: tuple[AudioAsset, ...]
    interaction_voices: tuple[AudioAsset, ...]
    warnings: tuple[str, ...]

    @property
    def live2d_available(self) -> bool:
        """是否存在通过 model3 文件引用校验的 Live2D 描述。"""

        return bool(self.live2d_models)

    @property
    def sprite_available(self) -> bool:
        return self.sprite_count > 0


def _inspect_wav(path: Path) -> AudioAsset:
    try:
        with wave.open(str(path), "rb") as stream:
            if stream.getnframes() <= 0 or stream.getframerate() <= 0:
                return AudioAsset(path, False, "WAV has no playable frames")
    except (OSError, EOFError, wave.Error):
        return AudioAsset(path, False, "WAV header is invalid")
    return AudioAsset(path, True)


def _file_is_nonempty(path: Path) -> bool:
    """判定文件是否含有真实资源数据，拒绝 Git LFS 指针占位文件。

    LFS 指针满足三行结构：``version https://git-lfs.github.com/spec/v1``、
    ``oid sha256:<hex>``、``size <n>``。若 ``git lfs pull`` 从未执行，
    仓库内只有数百字节的指针文本，把它当成权重会点亮 TTS 能力并让加载
    阶段在异步处失败；这里按缺失处理并记录 warning，不中断启动。
    """
    try:
        if not path.is_file():
            return False
        size = path.stat().st_size
        if size <= 0:
            return False
        if size <= 1024:
            with path.open("rb") as stream:
                head = stream.read(256)
            if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
                logger.warning("resource file is a Git LFS pointer: %s", path)
                return False
            lines = head.decode("utf-8", errors="replace").splitlines()
            if len(lines) == 3 and all(
                (
                    lines[0].startswith("version "),
                    lines[1].startswith("oid sha256:"),
                    lines[2].startswith("size "),
                )
            ):
                logger.warning("resource file is a Git LFS pointer: %s", path)
                return False
        return True
    except OSError:
        return False


def inspect_resources(root: str | Path) -> ResourceInventory:
    """探测模型、精灵、权重和音频，不因目录存在而假定能力可用。"""
    resource_root = Path(root).expanduser().resolve()
    warnings: list[str] = []
    if not resource_root.is_dir():
        return ResourceInventory(
            resource_root, (), 0, (), (), (), ("resource root is unavailable",)
        )

    model_root = resource_root / "live2d" / "model"
    model3_paths = tuple(sorted(model_root.rglob("*.model3.json")))
    legacy_model_paths = tuple(sorted(model_root.rglob("*.model.json")))
    model_paths = tuple(
        path for path in model3_paths if validate_model3_description(path) is not None
    )
    if legacy_model_paths:
        warnings.append("Live2D .model.json descriptors are unsupported without an exact schema")
    if model3_paths and len(model_paths) != len(model3_paths):
        warnings.append(
            "Live2D .model3.json descriptors are invalid; Live2D rendering is unavailable"
        )
    if not model_paths:
        warnings.append(
            "no usable Live2D model3 descriptor was found; Live2D rendering is unavailable"
        )

    sprite_paths = tuple((resource_root / "sprites").glob("*.webp"))
    sprite_count = sum(1 for path in sprite_paths if is_valid_webp(path))
    if sprite_paths and sprite_count != len(sprite_paths):
        warnings.append("invalid WebP sprite files were ignored")
    if not sprite_count:
        warnings.append("no WebP sprite resource was found for explicit sprite rendering")

    # 资源包当前使用 ``vits_models`` 目录；保留 ``models`` 作为显式兼容目录。
    # 只收集实际存在的权重文件，不能因目录名称存在就宣称 GSV 可用。
    weight_paths: set[Path] = set()
    for weights_root in (resource_root / "vits_models", resource_root / "models"):
        if weights_root.is_dir():
            weight_paths.update(weights_root.rglob("*.ckpt"))
            weight_paths.update(weights_root.rglob("*.pth"))
    weights = tuple(sorted(path.resolve() for path in weight_paths if _file_is_nonempty(path)))
    if not weights:
        warnings.append("no GPT-SoVITS weight file was found")

    reference_root = resource_root / "GPT-Sovits"
    references = tuple(_inspect_wav(path) for path in sorted(reference_root.rglob("*.wav")))
    if not any(asset.valid for asset in references):
        warnings.append("no valid GPT-SoVITS reference WAV was found")

    interaction_root = resource_root / "meapet" / "assets" / "interaction_voices"
    interactions = tuple(_inspect_wav(path) for path in sorted(interaction_root.rglob("*.wav")))
    if interactions and not any(asset.valid for asset in interactions):
        warnings.append("interaction WAV assets are invalid and must not be queued for playback")

    return ResourceInventory(
        root=resource_root,
        live2d_models=model_paths,
        sprite_count=sprite_count,
        gsv_weight_files=weights,
        reference_voices=references,
        interaction_voices=interactions,
        warnings=tuple(warnings),
    )
