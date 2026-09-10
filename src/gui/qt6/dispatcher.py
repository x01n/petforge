"""Qt 主线程调用调度器。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable
from typing import Any

try:  # Qt 依赖保持可选，核心服务仍可在无桌面环境运行。
    from PySide6.QtCore import QObject, Signal

    _QT_AVAILABLE = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    _QT_AVAILABLE = False


def _cancel_pending_futures(
    pending: set[concurrent.futures.Future[Any]],
) -> None:
    """取消待处理调用时使用稳定快照，允许完成回调同步修改原集合。"""

    for result in tuple(pending):
        result.cancel()


if _QT_AVAILABLE:

    class _InvocationBridge(QObject):
        invokeRequested = Signal(object)

        def __init__(self) -> None:
            super().__init__()
            self.invokeRequested.connect(self._run)

        def _run(self, payload: object) -> None:
            function, args, kwargs, result = payload
            if not isinstance(result, concurrent.futures.Future) or result.cancelled():
                return
            try:
                value = function(*args, **kwargs)
            except BaseException as exc:
                try:
                    result.set_exception(exc)
                except concurrent.futures.InvalidStateError:
                    return
            else:
                try:
                    result.set_result(value)
                except concurrent.futures.InvalidStateError:
                    return

    class QtMainThreadDispatcher:
        """将后台协程中的同步调用排入创建调度器的 Qt 线程。"""

        def __init__(self) -> None:
            self._thread_id = threading.get_ident()
            self._bridge = _InvocationBridge()
            self._pending: set[concurrent.futures.Future[Any]] = set()
            self._pending_lock = threading.Lock()
            self._closed = False

        def invoke(self, function: Callable[..., Any], *args: object, **kwargs: object) -> Any:
            if threading.get_ident() == self._thread_id:
                if self._closed:
                    raise RuntimeError("Qt dispatcher is closed")
                return function(*args, **kwargs)
            result: concurrent.futures.Future[Any] = concurrent.futures.Future()
            with self._pending_lock:
                if self._closed:
                    raise RuntimeError("Qt dispatcher is closed")
                self._pending.add(result)
            result.add_done_callback(self._discard_pending)
            try:
                self._bridge.invokeRequested.emit((function, args, kwargs, result))
            except BaseException:
                result.cancel()
                raise

            async def wait_result() -> Any:
                try:
                    return await asyncio.wrap_future(result)
                except asyncio.CancelledError:
                    result.cancel()
                    raise

            return wait_result()

        def _discard_pending(self, result: concurrent.futures.Future[Any]) -> None:
            with self._pending_lock:
                self._pending.discard(result)

        def close(self) -> None:
            with self._pending_lock:
                if self._closed:
                    return
                self._closed = True
                pending = tuple(self._pending)
                self._pending.clear()
            _cancel_pending_futures(set(pending))
            self._bridge.deleteLater()

else:

    class QtMainThreadDispatcher:  # pragma: no cover - 仅在无 Qt 环境使用
        """无 Qt 环境下的明确不可用调度器。"""

        def __init__(self) -> None:
            raise RuntimeError("PySide6 is not installed")


__all__ = ["QtMainThreadDispatcher"]
