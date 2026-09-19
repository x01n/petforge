"""TTS 后端的有界锁等待辅助：避免请求在无限期排队上永久挂起。"""

from __future__ import annotations

import asyncio

# 与超时相关常量的量纲协调：
# - timeout_seconds 是单次推理流从首个分片起每个空闲窗口的预算；
# - shutdown_timeout_seconds 是回收重试预算，通常远小于 runtime 查询预算。
# 锁等待与这两者无关，统一取 10s：既允许真实唤醒 CPU 的等待，也覆盖
# 绝大多数单次推理后退出的场景，且不超过任何一个公开的 60s 超时语义。
REQUEST_QUEUE_WAIT_SECONDS = 10.0


class QueueDepthProbe:
    """挂到锁对象上的可变计数，等待超时/取得锁时由边界快照读取。"""

    def __init__(self) -> None:
        self.value: object = 0

    # "waiting" 表示正在排队中，0 表示无排队。
    def __str__(self) -> str:
        return str(self.value)


class RequestBacklogError(RuntimeError):
    """请求等待独占锁超时，返回明确拒绝而不是永久排队。"""


async def wait_until_acquired(lock: asyncio.Lock, *, fallback_message: str) -> None:
    """有界等待一个锁，超时抛 RequestBacklogError 并留下排队标记。"""

    probe = getattr(lock, "queue_depth", None)
    if probe is not None:
        probe.value = "waiting"
    try:
        await asyncio.wait_for(lock.acquire(), timeout=REQUEST_QUEUE_WAIT_SECONDS)
    except TimeoutError:
        raise RequestBacklogError(fallback_message) from None
    if probe is not None:
        probe.value = 0
