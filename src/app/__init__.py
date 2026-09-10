from __future__ import annotations

__all__ = ["RuntimeLoop", "__version__"]

__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    """按需导入运行时线程封装，保持入口和帮助命令轻量。"""

    if name == "RuntimeLoop":
        from .loop import RuntimeLoop

        return RuntimeLoop
    raise AttributeError(name)
