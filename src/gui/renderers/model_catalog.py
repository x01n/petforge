from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from core.rendering.resources import (
    Live2DModelFiles,
    live2d_model_resource_version,
    validate_model3_description,
)

_MODEL_SUFFIX = ".model3.json"


@dataclass(frozen=True)
class Live2DModelEntry:
    """一个通过 ``model3`` 引用校验的模型。"""

    key: str
    descriptor: Path
    files: Live2DModelFiles
    resource_version: str = ""

    @property
    def relative_path(self) -> str:
        """返回配置中使用的资源根相对路径。"""

        return self.key

    @property
    def label(self) -> str:
        """返回不含 ``.model3.json`` 的用户可读名称。"""

        if self.key.endswith(_MODEL_SUFFIX):
            return self.key[: -len(_MODEL_SUFFIX)].rsplit("/", 1)[-1]
        return self.descriptor.stem


@dataclass(frozen=True)
class Live2DModelSelection:
    """一次模型选择的不可变结果。"""

    requested: str | None
    selected: Live2DModelEntry | None
    choices: tuple[str, ...]
    reason: str = ""

    @property
    def available(self) -> bool:
        """返回是否选中了可用模型。"""

        return self.selected is not None

    @property
    def explicit(self) -> bool:
        """返回用户是否填写了非空模型键。"""

        return bool(str(self.requested or "").strip()) and str(self.requested).strip() != "auto"

    @property
    def path(self) -> Path | None:
        """返回选中描述文件的绝对路径。"""

        return self.selected.descriptor if self.selected is not None else None

    @property
    def resource_version(self) -> str:
        """返回当前选择的内容版本，不暴露本地路径。"""

        return self.selected.resource_version if self.selected is not None else ""


class Live2DModelCatalog:
    """扫描并安全选择资源根目录内的 Live2D 模型。"""

    def __init__(self, resource_root: str | Path, entries: Iterable[Live2DModelEntry] = ()) -> None:
        self.resource_root = Path(resource_root).expanduser().resolve()
        self.model_root = (self.resource_root / "live2d" / "model").resolve()
        unique: dict[str, Live2DModelEntry] = {}
        for entry in entries:
            if not isinstance(entry, Live2DModelEntry):
                continue
            unique[entry.key] = entry
        self._entries = tuple(unique[key] for key in sorted(unique))

    @classmethod
    def scan(cls, resource_root: str | Path) -> Live2DModelCatalog:
        """扫描资源目录并仅保留通过完整引用校验的描述。"""

        root = Path(resource_root).expanduser().resolve()
        model_root = root / "live2d" / "model"
        entries: list[Live2DModelEntry] = []
        if not model_root.is_dir():
            return cls(root)
        for descriptor in sorted(model_root.rglob(f"*{_MODEL_SUFFIX}")):
            files = validate_model3_description(descriptor)
            if files is None:
                continue
            entry = cls._entry_from_files(root, files)
            if entry is not None:
                entries.append(entry)
        return cls(root, entries)

    @classmethod
    def from_descriptors(
        cls, resource_root: str | Path, descriptors: Iterable[str | Path]
    ) -> Live2DModelCatalog:
        """从已经扫描过的描述构造目录，同时再次执行边界校验。"""

        root = Path(resource_root).expanduser().resolve()
        entries: list[Live2DModelEntry] = []
        for descriptor in descriptors:
            try:
                path = Path(descriptor).expanduser().resolve()
                path.relative_to((root / "live2d" / "model").resolve())
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            files = validate_model3_description(path)
            if files is None:
                continue
            entry = cls._entry_from_files(root, files)
            if entry is not None:
                entries.append(entry)
        return cls(root, entries)

    @staticmethod
    def _entry_from_files(resource_root: Path, files: Live2DModelFiles) -> Live2DModelEntry | None:
        descriptor = files.descriptor.resolve()
        try:
            descriptor.relative_to((resource_root / "live2d" / "model").resolve())
            relative = descriptor.relative_to(resource_root).as_posix()
        except (ValueError, OSError):
            return None
        if not relative.endswith(_MODEL_SUFFIX):
            return None
        return Live2DModelEntry(
            relative,
            descriptor,
            files,
            live2d_model_resource_version(files),
        )

    @property
    def models(self) -> tuple[Live2DModelEntry, ...]:
        """返回按配置键排序的模型条目。"""

        return self._entries

    @property
    def choices(self) -> tuple[str, ...]:
        """返回可写入 ``rendering.model`` 的精确键。"""

        return tuple(item.key for item in self._entries)

    def select(self, requested: object = None) -> Live2DModelSelection:
        """解析模型键；空值/``auto`` 选择排序后的首个模型。"""

        choices = self.choices
        if requested is None or (isinstance(requested, str) and not requested.strip()):
            return Live2DModelSelection(
                requested if requested is None else "",
                self._entries[0] if self._entries else None,
                choices,
                "",
            )
        if not isinstance(requested, str):
            return Live2DModelSelection(
                str(requested),
                None,
                choices,
                "rendering.model must be a string or null",
            )
        value = requested.strip()
        if value == "auto":
            return Live2DModelSelection(
                value,
                self._entries[0] if self._entries else None,
                choices,
                "",
            )
        if not value or "\x00" in value or "\\" in value:
            return Live2DModelSelection(
                value,
                None,
                choices,
                "requested Live2D model path is invalid",
            )
        candidate = Path(value)
        if (
            candidate.is_absolute()
            or candidate.anchor
            or any(part == ".." for part in candidate.parts)
        ):
            return Live2DModelSelection(
                value,
                None,
                choices,
                "requested Live2D model path is outside the resource root",
            )
        normalized = candidate.as_posix()
        entry = next((item for item in self._entries if item.key == normalized), None)
        if entry is None:
            return Live2DModelSelection(
                value,
                None,
                choices,
                "requested Live2D model is not a usable model3 descriptor",
            )
        try:
            entry.descriptor.resolve().relative_to(self.model_root)
        except (OSError, RuntimeError, ValueError):
            return Live2DModelSelection(
                value,
                None,
                choices,
                "requested Live2D model path is outside the resource root",
            )
        return Live2DModelSelection(value, entry, choices, "")


# 便于调用方按简短名称导入，同时保留明确的 Live2D 类型名。
ModelCatalog = Live2DModelCatalog

__all__ = ["Live2DModelCatalog", "Live2DModelEntry", "Live2DModelSelection", "ModelCatalog"]
