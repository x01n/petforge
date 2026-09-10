"""可取消子进程及其子进程组的安全回收工具。"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from typing import Any


def process_group_spawn_kwargs() -> dict[str, object]:
    """返回创建独立进程组所需的 ``create_subprocess_exec`` 参数。"""

    if os.name == "posix":
        return {"start_new_session": True}
    raw_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", None)
    try:
        flags = int(raw_flags)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Windows process group creation is unavailable") from exc
    if flags <= 0:
        raise RuntimeError("Windows process group creation is unavailable")
    return {"creationflags": flags}


def _process_group_id(process: Any) -> int | None:
    if os.name != "posix":
        return None
    try:
        process_id = int(process.pid)
        group_id = os.getpgid(process_id)
    except (AttributeError, OSError, ProcessLookupError, TypeError, ValueError):
        return None
    return group_id if group_id == process_id else None


def process_group_id(process: Any) -> int | None:
    """在进程仍存活时保存其独立进程组 ID，供父进程提前退出后回收。"""

    return _process_group_id(process)


def close_process_transport(process: Any) -> None:
    """在事件循环仍存活时关闭 asyncio 子进程传输，避免析构阶段触碰已关闭循环。"""

    transport = getattr(process, "_transport", None)
    close = getattr(transport, "close", None)
    if callable(close):
        try:
            close()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass


def _process_identifier(process: Any) -> int | None:
    """读取进程 PID；无法确认时返回 ``None``。"""

    try:
        process_id = int(getattr(process, "pid"))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return process_id if process_id > 0 else None


def _windows_taskkill_tree(process_id: int, *, timeout: float) -> bool:
    """使用无 shell 的 Windows ``taskkill`` 回收仍存活进程及其子树。

    仅在根进程仍存活且 PID 已由当前 ``Popen``/asyncio 对象确认时调用，
    避免根进程退出后的 PID 复用误伤其他进程。
    """

    if os.name != "nt" or int(process_id) <= 0:
        return False
    try:
        limit = max(0.1, float(timeout))
    except (TypeError, ValueError, OverflowError):
        return False
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        result = subprocess.run(
            ("taskkill", "/PID", str(int(process_id)), "/T", "/F"),
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=limit,
            creationflags=creationflags,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        return False
    return int(getattr(result, "returncode", 1)) == 0


def _windows_control_break(process: Any) -> bool:
    """向 Windows 独立进程组发送 Ctrl+Break，不回退为根进程终止。"""

    if os.name != "nt":
        return False
    control_break = getattr(signal, "CTRL_BREAK_EVENT", None)
    send_signal = getattr(process, "send_signal", None)
    if control_break is None or not callable(send_signal):
        return False
    try:
        send_signal(control_break)
    except (OSError, ProcessLookupError, ValueError):
        return False
    return True


async def _terminate_windows_process_tree(process: Any, *, timeout: float) -> None:
    """按 Windows 的优雅信号、树回收、根进程回退顺序有界清理。"""

    if process is None:
        return
    process_id = _process_identifier(process)
    if getattr(process, "returncode", None) is not None:
        try:
            await process.wait()
        except (ProcessLookupError, ChildProcessError):
            pass
        return
    _windows_control_break(process)
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=max(0.1, float(timeout)))
        return
    except (TimeoutError, ProcessLookupError):
        pass
    if getattr(process, "returncode", None) is None:
        taskkill_ok = False
        if process_id is not None:
            taskkill_ok = await asyncio.to_thread(
                _windows_taskkill_tree,
                process_id,
                timeout=max(0.1, float(timeout)),
            )
        if not taskkill_ok:
            _signal_process_group(process, force=True)
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=max(0.1, float(timeout)))
    except (TimeoutError, ProcessLookupError):
        if getattr(process, "returncode", None) is None:
            _signal_process_group(process, force=True)
            try:
                await asyncio.wait_for(
                    asyncio.shield(process.wait()), timeout=min(0.5, max(0.1, float(timeout)))
                )
            except (TimeoutError, ProcessLookupError):
                pass


def terminate_process_sync(process: Any, *, timeout: float = 1.0) -> None:
    """同步 worker 使用与异步路径一致的有界进程树回收契约。"""

    if process is None or getattr(process, "returncode", None) is not None:
        return
    try:
        limit = max(0.1, float(timeout))
    except (TypeError, ValueError, OverflowError):
        limit = 1.0
    if os.name == "nt":
        _windows_control_break(process)
        try:
            process.wait(timeout=limit)
        except (subprocess.TimeoutExpired, ProcessLookupError, ChildProcessError):
            pass
        process_id = _process_identifier(process)
        taskkill_ok = (
            getattr(process, "returncode", None) is None
            and process_id is not None
            and _windows_taskkill_tree(process_id, timeout=limit)
        )
        if not taskkill_ok and getattr(process, "returncode", None) is None:
            _signal_process_group(process, force=True)
    else:
        _signal_process_group(process, force=False)
    try:
        process.wait(timeout=limit)
    except (subprocess.TimeoutExpired, ProcessLookupError, ChildProcessError):
        pass
    if getattr(process, "returncode", None) is None:
        _signal_process_group(process, force=True)
        try:
            process.wait(timeout=min(0.5, limit))
        except (subprocess.TimeoutExpired, ProcessLookupError, ChildProcessError):
            pass


def _signal_process_group(process: Any, *, force: bool, group_id: int | None = None) -> None:
    """优先向独立进程组发信号，失败时退回到主进程 API。"""

    if getattr(process, "returncode", None) is not None:
        if group_id is None:
            return
    if os.name == "posix":
        process_group = group_id if group_id is not None else _process_group_id(process)
        own_group = os.getpgrp()
        if process_group is not None and process_group != own_group:
            try:
                os.killpg(process_group, signal.SIGKILL if force else signal.SIGTERM)
                return
            except (OSError, ProcessLookupError):
                pass
    elif not force:
        control_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        send_signal = getattr(process, "send_signal", None)
        if control_break is not None and callable(send_signal):
            try:
                send_signal(control_break)
                return
            except (OSError, ProcessLookupError, ValueError):
                pass
    try:
        (process.kill if force else process.terminate)()
    except (OSError, ProcessLookupError):
        pass


async def terminate_process_tree(
    process: Any,
    *,
    timeout: float = 1.0,
    group_id: int | None = None,
) -> None:
    """终止并回收 worker 及其独立进程组，不阻塞调用方事件循环。"""

    if process is None:
        return
    normalized_timeout = max(0.1, float(timeout))
    if os.name == "nt":
        await _terminate_windows_process_tree(process, timeout=normalized_timeout)
        return
    if group_id is None:
        group_id = _process_group_id(process)
    if getattr(process, "returncode", None) is None:
        _signal_process_group(process, force=False, group_id=group_id)
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=normalized_timeout)
    except (TimeoutError, ProcessLookupError):
        pass
    if getattr(process, "returncode", None) is None:
        _signal_process_group(process, force=True, group_id=group_id)
    elif group_id is not None:
        # 主进程可能先退出而把子进程留在独立进程组中；再次发送 SIGKILL
        # 确保取消语义不会留下孤儿 worker。
        _signal_process_group(process, force=True, group_id=group_id)
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=normalized_timeout)
    except (TimeoutError, ProcessLookupError):
        return


__all__ = [
    "close_process_transport",
    "process_group_id",
    "process_group_spawn_kwargs",
    "terminate_process_sync",
    "terminate_process_tree",
]
