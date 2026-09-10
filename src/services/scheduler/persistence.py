from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class SchedulerStateStore(Protocol):
    """任务、触发器快照及执行记录的最小持久化契约。"""

    def load_tasks(self) -> Sequence[Mapping[str, Any]]:
        """返回可恢复的任务快照；损坏记录由实现过滤。"""

    def save_task(self, task: Mapping[str, Any]) -> None:
        """保存一个完整任务快照。"""

    def delete_task(self, task_id: str) -> None:
        """删除任务快照。"""

    def load_triggers(self) -> Sequence[Mapping[str, Any]]:
        """返回可恢复的触发器快照。"""

    def save_trigger(self, trigger: Mapping[str, Any]) -> None:
        """保存一个完整触发器快照。"""

    def delete_trigger(self, trigger_id: str) -> None:
        """删除触发器快照。"""

    def record_run(
        self,
        *,
        kind: str,
        item_id: str,
        owner: str,
        started_at: float,
        finished_at: float,
        status: str,
        error_text: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """追加一条有界执行记录。"""


__all__ = ["SchedulerStateStore"]
