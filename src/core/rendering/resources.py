from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Live2DModelFiles:
    """通过 model3 描述解析出的模型文件。"""

    descriptor: Path
    moc: Path
    textures: tuple[Path, ...]


def _iter_resource_references(value: object) -> tuple[str, ...]:
    """提取 ``FileReferences`` 中实际解析为文件的字符串。"""

    references: list[str] = []
    if isinstance(value, str):
        references.append(value)
    elif isinstance(value, list):
        for item in value:
            references.extend(_iter_resource_references(item))
    elif isinstance(value, dict):
        for item in value.values():
            references.extend(_iter_resource_references(item))
    return tuple(references)


def live2d_model_resource_version(files: Live2DModelFiles) -> str:
    """返回模型实际引用内容的稳定 SHA-256 版本。

    只读取描述文件目录内、由 ``FileReferences`` 指向且真实存在的文件；
    无法读取完整集合时返回空字符串，调用方必须把它视为不可比较版本。
    """

    descriptor = files.descriptor.resolve()
    paths = {descriptor, files.moc.resolve(), *(path.resolve() for path in files.textures)}
    try:
        value: Any = json.loads(descriptor.read_text(encoding="utf-8"))
        references = value.get("FileReferences") if isinstance(value, dict) else None
        for reference in _iter_resource_references(references):
            resolved = _resolve_child(descriptor.parent, reference)
            if resolved is not None:
                paths.add(resolved.resolve())
        if descriptor.name.endswith(".model3.json"):
            prefix = descriptor.name[: -len(".model3.json")]
            for suffix in (".actions.yaml", ".motionsync3.json"):
                sidecar = descriptor.with_name(f"{prefix}{suffix}")
                if _nonempty_file(sidecar):
                    paths.add(sidecar.resolve())
        digest = hashlib.sha256()
        for path in sorted(paths, key=lambda item: item.relative_to(descriptor.parent).as_posix()):
            relative = path.relative_to(descriptor.parent).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return ""


@dataclass(frozen=True)
class Live2DMotionSyncProfile:
    """经过严格字段校验的 Cubism ``motionsync3.json`` 口型配置。"""

    source: Path
    setting_id: str
    parameters: tuple[tuple[str, float, float], ...]
    audio_parameters: tuple[tuple[str, float, float, float, bool], ...]
    mappings: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    blend_ratio: float
    smoothing: float
    sample_rate: float

    @property
    def visemes(self) -> tuple[str, ...]:
        """返回按资源顺序校验过的音素标签。"""

        return tuple(identifier for identifier, *_rest in self.audio_parameters)

    def as_mapping(self) -> dict[str, object]:
        """转换为可安全注入 Web 页面、且不包含本地路径的映射。"""

        return {
            "version": 1,
            "setting_id": self.setting_id,
            "parameters": [
                {"id": identifier, "min": minimum, "max": maximum}
                for identifier, minimum, maximum in self.parameters
            ],
            "audio_parameters": [
                {
                    "id": identifier,
                    "min": minimum,
                    "max": maximum,
                    "scale": scale,
                    "enabled": enabled,
                }
                for identifier, minimum, maximum, scale, enabled in self.audio_parameters
            ],
            "mappings": {
                identifier: {
                    "targets": [{"id": target_id, "value": value} for target_id, value in targets]
                }
                for identifier, targets in self.mappings
            },
            "post_processing": {
                "blend_ratio": self.blend_ratio,
                "smoothing": self.smoothing,
                "sample_rate": self.sample_rate,
            },
        }


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _resolve_child(parent: Path, reference: object) -> Path | None:
    if not isinstance(reference, str) or not reference.strip():
        return None
    reference_path = Path(reference)
    if reference_path.is_absolute() or reference_path.anchor or "\x00" in reference:
        return None
    try:
        resolved = (parent / reference_path).resolve()
        resolved.relative_to(parent)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if _nonempty_file(resolved) else None


def validate_model3_description(path: str | Path) -> Live2DModelFiles | None:
    """校验当前 Web Live2D 页面使用的 model3 字段和文件引用。

    只接受 ``FileReferences.Moc`` 非空字符串、``Textures`` 非空字符串数组，
    并要求每个引用都解析到描述文件目录内的非空普通文件。``.model.json``
    没有在当前项目中定义精确字段契约，由调用方单独标记为不支持。
    """

    try:
        descriptor = Path(path).expanduser().resolve()
        value: Any = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    references = value.get("FileReferences")
    if not isinstance(references, dict):
        return None
    moc_reference = references.get("Moc")
    texture_references = references.get("Textures")
    if not isinstance(moc_reference, str) or not moc_reference.strip():
        return None
    if not isinstance(texture_references, list) or not texture_references:
        return None
    moc_path = _resolve_child(descriptor.parent, moc_reference)
    if moc_path is None:
        return None
    texture_paths: list[Path] = []
    for reference in texture_references:
        texture_path = _resolve_child(descriptor.parent, reference)
        if texture_path is None:
            return None
        texture_paths.append(texture_path)
    return Live2DModelFiles(descriptor, moc_path, tuple(texture_paths))


def _read_json_object(path: str | Path) -> dict[str, Any] | None:
    """读取非空 JSON 对象；失败时返回空值。"""

    try:
        value: Any = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finite_number(value: object) -> bool:
    """判断 JSON 数字是否可安全交给 Cubism Core。"""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _nonnegative_integer(value: object) -> bool:
    """判断 JSON 计数是否为非负整数。"""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


_MOTION_SYNC_VISEMES = ("Silence", "A", "I", "U", "E", "O")


def validate_motionsync3_description(
    path: str | Path,
) -> Live2DMotionSyncProfile | None:
    """读取并严格校验当前运行时支持的 Mouth MotionSync 字段。

    当前接入只消费资源中明确出现的 ``Shape`` 映射，以及
    ``Silence/A/I/U/E/O`` 六个音素标签。任何重复 ID、缺失标签、越界
    参数值或不完整后处理字段都会拒绝整份配置，调用方继续使用现有
    overlay 回退，不会把未验证数值写入 Cubism Core。
    """

    try:
        source = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    value = _read_json_object(source)
    version = value.get("Version") if value is not None else None
    if value is None or not isinstance(version, int) or isinstance(version, bool) or version != 1:
        return None
    meta = value.get("Meta")
    settings = value.get("Settings")
    if not isinstance(meta, dict) or not isinstance(settings, list) or not settings:
        return None
    setting_count = meta.get("SettingCount")
    dictionary = meta.get("Dictionary")
    if (
        not _nonnegative_integer(setting_count)
        or setting_count != len(settings)
        or not isinstance(dictionary, list)
        or len(dictionary) != setting_count
    ):
        return None

    dictionary_ids: set[str] = set()
    for entry in dictionary:
        if not isinstance(entry, dict):
            return None
        identifier = entry.get("Id")
        name = entry.get("Name")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in dictionary_ids
            or not isinstance(name, str)
            or not name.strip()
        ):
            return None
        dictionary_ids.add(identifier)

    mouth_settings = [
        setting
        for setting in settings
        if isinstance(setting, dict) and setting.get("UseCase") == "Mouth"
    ]
    if len(mouth_settings) != 1:
        return None
    setting = mouth_settings[0]
    setting_id = setting.get("Id")
    if (
        not isinstance(setting_id, str)
        or not setting_id.strip()
        or setting_id not in dictionary_ids
        or setting.get("AnalysisType") != "CRI"
    ):
        return None

    raw_parameters = setting.get("CubismParameters")
    if not isinstance(raw_parameters, list) or not raw_parameters:
        return None
    parameters: list[tuple[str, float, float]] = []
    parameter_ranges: dict[str, tuple[float, float]] = {}
    for item in raw_parameters:
        if not isinstance(item, dict):
            return None
        identifier = item.get("Id")
        minimum = item.get("Min")
        maximum = item.get("Max")
        damper = item.get("Damper")
        smooth = item.get("Smooth")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in parameter_ranges
            or not _finite_number(minimum)
            or not _finite_number(maximum)
            or float(minimum) > float(maximum)
            or not _finite_number(damper)
            or float(damper) < 0
            or not _finite_number(smooth)
            or float(smooth) < 0
        ):
            return None
        limits = (float(minimum), float(maximum))
        parameter_ranges[identifier] = limits
        parameters.append((identifier, *limits))

    raw_audio = setting.get("AudioParameters")
    if not isinstance(raw_audio, list) or not raw_audio:
        return None
    audio_parameters: list[tuple[str, float, float, float, bool]] = []
    audio_ids: set[str] = set()
    for item in raw_audio:
        if not isinstance(item, dict):
            return None
        identifier = item.get("Id")
        minimum = item.get("Min")
        maximum = item.get("Max")
        scale = item.get("Scale")
        enabled = item.get("Enabled")
        if (
            not isinstance(identifier, str)
            or identifier not in _MOTION_SYNC_VISEMES
            or identifier in audio_ids
            or not _finite_number(minimum)
            or not _finite_number(maximum)
            or float(minimum) > float(maximum)
            or not _finite_number(scale)
            or float(scale) < 0
            or not isinstance(enabled, bool)
        ):
            return None
        audio_ids.add(identifier)
        audio_parameters.append((identifier, float(minimum), float(maximum), float(scale), enabled))
    if audio_ids != set(_MOTION_SYNC_VISEMES):
        return None

    raw_mappings = setting.get("Mappings")
    if not isinstance(raw_mappings, list) or not raw_mappings:
        return None
    mappings: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    mapping_ids: set[str] = set()
    for item in raw_mappings:
        if not isinstance(item, dict) or item.get("Type") != "Shape":
            return None
        identifier = item.get("Id")
        raw_targets = item.get("Targets")
        if (
            not isinstance(identifier, str)
            or identifier not in audio_ids
            or identifier in mapping_ids
            or not isinstance(raw_targets, list)
            or not raw_targets
        ):
            return None
        targets: list[tuple[str, float]] = []
        target_ids: set[str] = set()
        for target in raw_targets:
            if not isinstance(target, dict):
                return None
            target_id = target.get("Id")
            target_value = target.get("Value")
            if (
                not isinstance(target_id, str)
                or target_id not in parameter_ranges
                or target_id in target_ids
                or not _finite_number(target_value)
            ):
                return None
            minimum, maximum = parameter_ranges[target_id]
            resolved_value = float(target_value)
            if resolved_value < minimum or resolved_value > maximum:
                return None
            target_ids.add(target_id)
            targets.append((target_id, resolved_value))
        mapping_ids.add(identifier)
        mappings.append((identifier, tuple(targets)))
    if mapping_ids != audio_ids:
        return None

    post_processing = setting.get("PostProcessing")
    if not isinstance(post_processing, dict):
        return None
    blend_ratio = post_processing.get("BlendRatio")
    smoothing = post_processing.get("Smoothing")
    sample_rate = post_processing.get("SampleRate")
    if (
        not _finite_number(blend_ratio)
        or not 0 <= float(blend_ratio) <= 1
        or not _finite_number(smoothing)
        or float(smoothing) < 0
        or not _finite_number(sample_rate)
        or float(sample_rate) <= 0
    ):
        return None
    return Live2DMotionSyncProfile(
        source=source,
        setting_id=setting_id,
        parameters=tuple(parameters),
        audio_parameters=tuple(audio_parameters),
        mappings=tuple(mappings),
        blend_ratio=float(blend_ratio),
        smoothing=float(smoothing),
        sample_rate=float(sample_rate),
    )


def validate_expression3_description(path: str | Path) -> bool:
    """校验 Cubism exp3.json 的实际表达式字段。

    只接受 Cubism ExpressionMotion 会读取的参数数组；空对象、空文件和
    任意占位 JSON 不会被当作可加载表情。Blend 遵循 Core 支持的
    Add、Multiply、Overwrite 三种值，省略时由 Core 使用 Add。
    """

    value = _read_json_object(path)
    if value is None:
        return False
    parameters = value.get("Parameters")
    if not isinstance(parameters, list) or not parameters:
        return False
    for parameter in parameters:
        if not isinstance(parameter, dict):
            return False
        identifier = parameter.get("Id")
        if not isinstance(identifier, str) or not identifier.strip():
            return False
        if not _finite_number(parameter.get("Value")):
            return False
        blend = parameter.get("Blend")
        if blend is not None and blend not in {"Add", "Multiply", "Overwrite"}:
            return False
    for key in ("FadeInTime", "FadeOutTime"):
        if key in value and not _finite_number(value[key]):
            return False
    return True


def _validate_motion_curve_segments(segments: object) -> tuple[int, int] | None:
    """校验 Cubism motion 曲线的扁平分段编码并返回段/点数量。"""

    if not isinstance(segments, list) or len(segments) < 5:
        return None
    if any(not _finite_number(item) for item in segments):
        return None
    position = 2
    segment_count = 0
    point_count = 1
    while position < len(segments):
        segment_type = segments[position]
        if (
            not isinstance(segment_type, int)
            or isinstance(segment_type, bool)
            or segment_type not in {0, 1, 2, 3}
        ):
            return None
        width = 7 if segment_type == 1 else 3
        if position + width > len(segments):
            return None
        point_count += 3 if segment_type == 1 else 1
        segment_count += 1
        position += width
    if position != len(segments):
        return None
    return segment_count, point_count


def validate_motion3_description(path: str | Path) -> bool:
    """校验 Cubism motion3.json 的 Meta/Curves/UserData 契约。"""

    value = _read_json_object(path)
    if value is None:
        return False
    meta = value.get("Meta")
    curves = value.get("Curves")
    if not isinstance(meta, dict) or not isinstance(curves, list) or not curves:
        return False
    duration = meta.get("Duration")
    fps = meta.get("Fps")
    if not _finite_number(duration) or float(duration) < 0:
        return False
    if not _finite_number(fps) or float(fps) <= 0:
        return False
    if "Loop" in meta and not isinstance(meta["Loop"], bool):
        return False
    curve_count = meta.get("CurveCount")
    total_segment_count = meta.get("TotalSegmentCount")
    total_point_count = meta.get("TotalPointCount")
    user_data_count = meta.get("UserDataCount")
    total_user_data_size = meta.get("TotalUserDataSize")
    if not all(
        _nonnegative_integer(item)
        for item in (
            curve_count,
            total_segment_count,
            total_point_count,
            user_data_count,
            total_user_data_size,
        )
    ):
        return False
    if curve_count != len(curves) or curve_count == 0:
        return False
    actual_segments = 0
    actual_points = 0
    for curve in curves:
        if not isinstance(curve, dict):
            return False
        target = curve.get("Target")
        if target not in {"Model", "Parameter", "PartOpacity"}:
            return False
        identifier = curve.get("Id")
        if not isinstance(identifier, str) or not identifier.strip():
            return False
        for key in ("FadeInTime", "FadeOutTime"):
            if key in curve and not _finite_number(curve[key]):
                return False
        counts = _validate_motion_curve_segments(curve.get("Segments"))
        if counts is None:
            return False
        actual_curve_segments, actual_curve_points = counts
        actual_segments += actual_curve_segments
        actual_points += actual_curve_points
    if actual_segments != total_segment_count or actual_points != total_point_count:
        return False
    if user_data_count:
        user_data = value.get("UserData")
        if not isinstance(user_data, list) or len(user_data) != user_data_count:
            return False
        for event in user_data:
            if not isinstance(event, dict):
                return False
            if not _finite_number(event.get("Time")):
                return False
            if not isinstance(event.get("Value"), str):
                return False
    return True


def is_valid_webp(path: str | Path) -> bool:
    """验证 WebP 文件的 RIFF/WEBP 容器头。"""

    value = Path(path)
    try:
        if not value.is_file() or value.stat().st_size < 12:
            return False
        with value.open("rb") as stream:
            header = stream.read(12)
    except OSError:
        return False
    return header[:4] == b"RIFF" and header[8:12] == b"WEBP"


__all__ = [
    "Live2DModelFiles",
    "Live2DMotionSyncProfile",
    "is_valid_webp",
    "live2d_model_resource_version",
    "validate_expression3_description",
    "validate_model3_description",
    "validate_motion3_description",
    "validate_motionsync3_description",
]
