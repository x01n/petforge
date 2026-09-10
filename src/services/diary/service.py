"""公开投影与内部受控桌宠日记服务。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from db.database import Database
from db.pet_diary_repository import (
    MAX_DIARY_AUDIT_ROWS,
    PRIVATE_VISIBILITY,
    PetDiaryAuditRecord,
    PetDiaryEntry,
    PetDiaryRepository,
)

_UNSET = object()


class PetDiaryInternalSession:
    """绑定明确审计身份的内部日记能力；实例不应进入 UI 或 Web 投影。"""

    def __init__(self, repository: PetDiaryRepository, actor: object) -> None:
        self._repository = repository
        self._access = repository._issue_internal_access(actor)

    @property
    def actor(self) -> str:
        """返回写入审计表的稳定调用方身份。"""

        return self._access.actor

    def create(
        self,
        content: object,
        *,
        visibility: object = PRIVATE_VISIBILITY,
        priority: object = 5,
        tags: Sequence[object] = (),
        retention_until: object = None,
    ) -> PetDiaryEntry:
        """创建日记；默认私密，公开必须显式指定。"""

        return self._repository._create(
            self._access,
            content,
            visibility=visibility,
            priority=priority,
            tags=tags,
            retention_until=retention_until,
        )

    def get(self, entry_id: object, *, include_archived: bool = True) -> PetDiaryEntry | None:
        """读取单条公开或私密日记，并留下读取审计。"""

        return self._repository._get(
            self._access,
            entry_id,
            include_archived=bool(include_archived),
        )

    def list_entries(
        self,
        *,
        include_private: bool = False,
        include_archived: bool = False,
        limit: int = 100,
    ) -> tuple[PetDiaryEntry, ...]:
        """列出日记；私密内容必须由调用方显式开启。"""

        return self._repository._list(
            self._access,
            include_private=bool(include_private),
            include_archived=bool(include_archived),
            limit=limit,
        )

    def update(
        self,
        entry_id: object,
        *,
        content: object = _UNSET,
        visibility: object = _UNSET,
        priority: object = _UNSET,
        tags: Sequence[object] | object = _UNSET,
        retention_until: object = _UNSET,
    ) -> PetDiaryEntry | None:
        """更新指定字段，允许用 ``None`` 明确清除保留期限。"""

        supplied = {
            "content": content,
            "visibility": visibility,
            "priority": priority,
            "tags": tags,
            "retention_until": retention_until,
        }
        changes = {name: value for name, value in supplied.items() if value is not _UNSET}
        return self._repository._update(self._access, entry_id, changes)

    def set_archived(self, entry_id: object, *, archived: bool = True) -> PetDiaryEntry | None:
        """归档或恢复条目；归档条目不会进入默认列表。"""

        return self._repository._set_archived(
            self._access,
            entry_id,
            archived=bool(archived),
        )

    def delete(self, entry_id: object, *, force: bool = False) -> bool:
        """永久删除条目；保留期限内必须显式使用 ``force``。"""

        return self._repository._delete(self._access, entry_id, force=bool(force))

    def archive_expired(self) -> tuple[int, ...]:
        """归档已超过 retention_until 的条目，并留下逐条审计。"""

        return self._repository._archive_expired(self._access)

    def audit_log(
        self,
        *,
        limit: int = 100,
        entry_id: object | None = None,
    ) -> tuple[PetDiaryAuditRecord, ...]:
        """读取不含正文和标签的日记审计记录。"""

        return self._repository._list_audit(
            self._access,
            limit=limit,
            entry_id=entry_id,
        )


class PetDiaryService:
    """隔离普通公开读取与桌宠内部读写能力。"""

    def __init__(
        self,
        database: Database,
        *,
        clock: Callable[[], float] | None = None,
        max_audit_rows: int = MAX_DIARY_AUDIT_ROWS,
    ) -> None:
        self._repository = PetDiaryRepository(
            database,
            clock=clock,
            max_audit_rows=max_audit_rows,
        )

    def open_internal_session(self, actor: object) -> PetDiaryInternalSession:
        """为内部 Agent 或维护流程建立带身份审计的能力会话。"""

        return PetDiaryInternalSession(self._repository, actor)

    def list_entries(self, *, limit: int = 100) -> tuple[PetDiaryEntry, ...]:
        """返回公开且未归档条目，不能通过参数提升到私密读取。"""

        return self._repository.list_public(limit=limit)

    def get_entry(self, entry_id: object) -> PetDiaryEntry | None:
        """读取公开且未归档条目。"""

        return self._repository.get_public(entry_id)

    def status(self) -> dict[str, object]:
        """返回可安全进入普通状态投影的公开统计。"""

        return {
            "enabled": True,
            "public_entry_count": self._repository.public_count(),
        }


__all__ = ["PetDiaryInternalSession", "PetDiaryService"]
