from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from .runtime import ApplicationRuntime

ResultT = TypeVar("ResultT")
logger = logging.getLogger(__name__)


class RuntimeLoop:
    """在专用 asyncio 线程中托管应用运行时，并向 Qt 提供提交边界。"""

    def __init__(self, runtime: ApplicationRuntime) -> None:
        self.runtime = runtime
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        # stop() 可能在后台线程完成启动前被调用；使用独立标志让线程在
        # run_until_complete 返回后直接进入收尾，而不是遗留一个永久事件循环。
        self._stop_requested = threading.Event()
        self._startup_task: asyncio.Task[object] | None = None
        self._close_lock = threading.Lock()
        self._close_submitted = False
        self._startup_error: BaseException | None = None
        self._closed = False

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive() and self._loop is not None)

    def start(self, *, timeout: float = 10.0) -> None:

        if self.running:
            return
        if self._closed:
            raise RuntimeError("runtime loop is closed")
        self._ready.clear()
        self._stop_requested.clear()
        with self._close_lock:
            self._close_submitted = False
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name="meapet-runtime",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(max(0.1, float(timeout))):
            self.stop(timeout=timeout)
            raise TimeoutError("runtime loop did not start in time")
        if self._startup_error is not None:
            error = self._startup_error
            self.stop(timeout=timeout)
            raise RuntimeError("runtime background services failed to start") from error

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            startup = loop.create_task(self.runtime.start_background())
            self._startup_task = startup
            loop.run_until_complete(startup)
            self._ready.set()
            if not self._stop_requested.is_set():
                loop.run_forever()
        except BaseException as exc:  # 启动错误需要交给主线程处理
            self._startup_error = exc
            self._ready.set()
        finally:
            # 启动阶段失败或宿主未能提交 close 时，仍在运行时线程内完整释放
            # 对话、路由、TTS 和数据库资源，避免只关闭 SQLite。
            with self._close_lock:
                close_owned = not self._close_submitted
                if close_owned:
                    self._close_submitted = True
            if close_owned:
                try:
                    loop.run_until_complete(self.runtime.close())
                except BaseException as exc:
                    logger.warning(
                        "runtime close during loop teardown failed: %s", type(exc).__name__
                    )
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self._loop = None
            self._startup_task = None
            if not self._ready.is_set():
                # 即使初始化阶段抛出未预期异常，start() 也必须被唤醒并把错误
                # 交给调用方，不能让主线程永久等待 ready 事件。
                self._ready.set()

    def submit(self, coroutine: Coroutine[Any, Any, ResultT]) -> concurrent.futures.Future[ResultT]:

        loop = self._loop
        if not self.running or loop is None:
            coroutine.close()
            raise RuntimeError("runtime loop is not running")
        try:
            return asyncio.run_coroutine_threadsafe(coroutine, loop)
        except RuntimeError:
            # 事件循环可能在 running 检查后刚好关闭，必须回收未提交的协程对象。
            coroutine.close()
            raise

    def stop(self, *, timeout: float = 10.0) -> None:
        if self._closed:
            return

        thread = self._thread
        if thread is None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                try:
                    asyncio.run(self.runtime.close())
                except BaseException as exc:
                    logger.warning("runtime close failed: %s", type(exc).__name__)
            else:
                # 该同步 API 不能阻塞当前事件循环；至少保证数据库句柄不泄漏。
                self.runtime.database.close()
            self._closed = True
            return
        if thread is threading.current_thread():
            raise RuntimeError("RuntimeLoop.stop must be called outside the runtime thread")

        # 等待启动阶段结束后再向事件循环提交 close；否则 stop 可能看到
        # loop.is_running() 为 False，随后 join 阻塞，而后台线程刚好进入
        # run_forever，导致 Ctrl+C 无法完成退出。
        wait_timeout = max(0.1, float(timeout))
        self._stop_requested.set()
        startup_task = self._startup_task
        loop = self._loop
        if startup_task is not None and loop is not None and not self._ready.is_set():
            try:
                loop.call_soon_threadsafe(startup_task.cancel)
            except RuntimeError:
                pass
        self._ready.wait(wait_timeout)
        loop = self._loop
        if loop is not None and loop.is_running():
            close_future: concurrent.futures.Future[None] | None = None
            with self._close_lock:
                if not self._close_submitted:
                    close_coroutine = self.runtime.close()
                    try:
                        close_future = asyncio.run_coroutine_threadsafe(close_coroutine, loop)
                    except RuntimeError:
                        close_coroutine.close()
                    else:
                        self._close_submitted = True
            if close_future is not None:
                try:
                    close_future.result(timeout=max(0.1, float(timeout)))
                except BaseException as exc:
                    logger.warning("runtime close failed: %s", type(exc).__name__)
                    close_future.cancel()
                    # 让线程 teardown 在提交的关闭协程超时或失败时接管
                    # 一次兜底 close，避免只停止事件循环而遗漏服务释放。
                    with self._close_lock:
                        self._close_submitted = False
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except RuntimeError:
                    pass
        thread.join(max(0.1, float(timeout)))
        if thread.is_alive():
            raise TimeoutError("runtime loop did not stop in time")
        self.runtime.database.close()
        self._thread = None
        self._closed = True


__all__ = ["RuntimeLoop"]
