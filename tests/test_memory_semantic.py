import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import services.memory.semantic as semantic_module
from core.memory_semantic import (
    PersistentHnswIndex,
    SemanticMemoryError,
    SemanticModelSpec,
    semantic_worker_environment,
)
from db import Database
from services.memory import MemoryService
from services.memory.semantic import SemanticMemoryController, SemanticMemorySettings


def _settings(root: Path, *, revision: str = "r1", index_dir: Path | None = None) -> dict:
    return {
        "semantic": {
            "enabled": True,
            "provider": "sentence_transformers",
            "model_path": str(root),
            "model_id": "local-test-model",
            "model_revision": revision,
            "dimension": 8,
            "batch_size": 2,
            "timeout_seconds": 1.0,
            "startup_timeout_seconds": 1.0,
            "shutdown_timeout_seconds": 0.5,
            "index_dir": str(index_dir) if index_dir is not None else "",
        }
    }


class _FakeProvider:
    def __init__(self, revision: str = "r1", package_version: str = "fake-1") -> None:
        self.spec = SemanticModelSpec(
            "sentence_transformers",
            package_version,
            "local-test-model",
            revision,
            8,
        )
        self.closed = False

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        values = [0.0] * 8
        values[0] = 1.0 if "茶" in text or "tea" in text else 0.1
        values[1] = 1.0 if "猫" in text or "cat" in text else 0.1
        values[2] = 1.0
        return tuple(values)

    def embed_documents(self, texts, *, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            raise SemanticMemoryError("cancelled")
        return tuple(self._vector(str(text)) for text in texts)

    def embed_query(self, text, *, cancel_event=None):
        return self._vector(str(text))

    def close(self):
        self.closed = True


class _BlockingProvider(_FakeProvider):
    """忽略请求取消，用于证明异步边界不依赖替身合作。"""

    def __init__(self) -> None:
        super().__init__()
        self._gate = threading.Lock()
        self.release = threading.Event()
        self.document_started = threading.Event()
        self.query_started = threading.Event()
        self.block_documents = False
        self.block_query = False

    def embed_documents(self, texts, *, cancel_event=None):
        with self._gate:
            if self.block_documents:
                self.document_started.set()
                self.release.wait()
            return tuple(self._vector(str(text)) for text in texts)

    def embed_query(self, text, *, cancel_event=None):
        with self._gate:
            if self.block_query:
                self.query_started.set()
                self.release.wait()
            return self._vector(str(text))

    def close(self):
        self.release.set()
        super().close()


class _CancelAwareProvider(_FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_seen = threading.Event()

    def embed_query(self, text, *, cancel_event=None):
        del text
        assert cancel_event is not None
        while not cancel_event.is_set():
            time.sleep(0.005)
        self.cancel_seen.set()
        raise SemanticMemoryError("cancelled")


def test_default_semantic_settings_are_disabled_without_model_or_network(tmp_path):
    service = MemoryService(Database(tmp_path / "default.sqlite3"))
    try:
        status = service.status()
        assert status["semantic_enabled"] is False
        assert status["semantic_index"] == "disabled"
        assert status["sparse_index_kind"] == "deterministic_hash_lexical"
    finally:
        service.close()


def test_semantic_status_redacts_path_and_secret_like_model_identity(tmp_path):
    service = MemoryService(
        Database(tmp_path / "redaction.sqlite3"),
        settings={
            "semantic": {
                "enabled": False,
                "model_id": "sk-secret-value",
                "model_revision": "/home/user/private-revision",
            }
        },
    )
    try:
        status = service.status()
        assert str(status["semantic_model_id"]).startswith("sha256:")
        assert str(status["semantic_model_revision"]).startswith("sha256:")
        assert "secret" not in repr(status).lower()
        assert "/home/user" not in repr(status)
    finally:
        service.close()


def test_inactive_semantic_cleanup_never_deletes_external_or_symlink_targets(tmp_path):
    database = Database(tmp_path / "cleanup.sqlite3")
    controller = SemanticMemoryController(
        database,
        SemanticMemorySettings(enabled=False),
        capacity=100,
    )
    index_dir = tmp_path / "cleanup.sqlite3.semantic-index"
    index_dir.mkdir()
    victim = tmp_path / "victim.hnsw"
    victim.write_bytes(b"keep")
    link = index_dir / ("a" * 32 + ".hnsw")
    link.symlink_to(victim)
    try:
        with database.transaction() as connection:
            connection.execute(
                "INSERT INTO memory_ann_generations "
                "(generation, settings_fingerprint, provider, package_version, model_id, "
                "model_revision, dimension, index_revision, index_path, item_count, "
                "applied_sequence, created_at, active) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    "b" * 32,
                    "fingerprint",
                    "sentence_transformers",
                    "fake",
                    "model",
                    "r1",
                    8,
                    "a" * 32,
                    str(link),
                    0,
                    0,
                    time.time(),
                ),
            )
        controller._cleanup_inactive_generations(
            "c" * 32,
            SemanticMemorySettings(enabled=False, index_dir=str(index_dir)),
        )
        assert victim.exists()
        assert link.exists()
    finally:
        controller.close()
        database.close()


def test_worker_environment_does_not_forward_secret(monkeypatch):
    source = dict(os.environ)
    source["MEAPET_TEST_SECRET"] = "must-not-cross-process"
    environment = semantic_worker_environment(source)
    assert "MEAPET_TEST_SECRET" not in environment
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.memory_semantic_worker",
            "--audit-environment",
            "MEAPET_TEST_SECRET",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    assert json.loads(result.stdout)["present"] is False


def test_hnsw_upsert_replaces_existing_active_label() -> None:
    class _Index:
        max_elements = 16

        def __init__(self) -> None:
            self.deleted: list[int] = []
            self.added: list[dict[str, object]] = []

        def mark_deleted(self, identity: int) -> None:
            self.deleted.append(identity)

        def add_items(self, vectors, ids, **kwargs) -> None:
            self.added.append({"vectors": vectors, "ids": ids, "kwargs": kwargs})

    fake_index = _Index()
    index = object.__new__(PersistentHnswIndex)
    index.dimension = 3
    index._index = fake_index
    index._active_ids = {11}
    index._lock = threading.RLock()

    index.upsert(11, (0.0, 1.0, 0.0))

    assert fake_index.deleted == [11]
    assert fake_index.added[-1]["ids"] == [11]
    assert fake_index.added[-1]["kwargs"]["replace_deleted"] is True
    assert index.active_ids == frozenset({11})


def test_hnsw_index_persists_and_removes_tombstoned_id(tmp_path):
    pytest.importorskip("hnswlib")
    index = PersistentHnswIndex.build(
        ((11, (1.0, 0.0, 0.0)), (22, (0.0, 1.0, 0.0))),
        dimension=3,
        capacity=16,
        m=8,
        ef_construction=32,
        ef_search=16,
    )
    index.save_revision(tmp_path, "a" * 32)
    restored = PersistentHnswIndex.load(
        tmp_path / ("a" * 32 + ".hnsw"),
        active_ids=(11, 22),
        dimension=3,
        capacity=16,
        m=8,
        ef_construction=32,
        ef_search=16,
    )
    assert restored.search((1.0, 0.0, 0.0), limit=2)[0][0] == 11
    restored.remove(11)
    assert all(memory_id != 11 for memory_id, _ in restored.search((1.0, 0.0, 0.0), limit=2))


def test_semantic_controller_batches_mutations_and_persists_tombstone(tmp_path, monkeypatch):
    pytest.importorskip("hnswlib")
    providers = []

    def factory(settings, *, cancel_event):
        provider = _FakeProvider(settings.model_revision)
        providers.append(provider)
        return provider

    monkeypatch.setattr(semantic_module, "create_dense_provider", factory)
    path = tmp_path / "memory.sqlite3"
    database = Database(path)
    service = MemoryService(database, settings=_settings(tmp_path))
    try:
        assert service.wait_for_semantic_idle(5)
        assert service.status()["semantic_index"] == "ready"
        memory = service.create("用户喜欢猫和茶")
        assert service.wait_for_semantic_idle(5)
        assert service.status()["semantic_index_size"] == 1
        assert service.search("tea", limit=1)[0].id == memory.id

        service.update(memory.id, content="用户改为喜欢狗和水")
        assert service.wait_for_semantic_idle(5)
        service.delete(memory.id)
        assert service.wait_for_semantic_idle(5)
        row = database.connection.execute(
            "SELECT tombstone FROM memory_semantic_vectors WHERE memory_id = ?", (memory.id,)
        ).fetchone()
        assert row is not None and int(row[0]) == 1
        assert service.status()["semantic_pending_mutations"] == 0
    finally:
        service.close()
    assert providers and all(provider.closed for provider in providers)


def test_revision_failure_keeps_previous_active_generation(tmp_path, monkeypatch):
    pytest.importorskip("hnswlib")
    state = {"fail": False}

    def factory(settings, *, cancel_event):
        if state["fail"]:
            raise SemanticMemoryError("model_load_failed")
        return _FakeProvider(settings.model_revision)

    monkeypatch.setattr(semantic_module, "create_dense_provider", factory)
    database = Database(tmp_path / "rollback.sqlite3")
    service = MemoryService(database, settings=_settings(tmp_path, revision="r1"))
    try:
        assert service.wait_for_semantic_idle(5)
        memory = service.create("用户喜欢茶")
        assert service.wait_for_semantic_idle(5)
        before = service.status()["semantic_active_generation"]
        assert before
        state["fail"] = True
        service.configure(_settings(tmp_path, revision="r2"))
        assert service.wait_for_semantic_idle(5)
        status = service.status()
        assert status["semantic_index"] == "degraded"
        assert status["semantic_active_generation"] == before
        active = database.connection.execute(
            "SELECT generation, active FROM memory_ann_generations WHERE active = 1"
        ).fetchone()
        assert active is not None and active[0] == before and int(active[1]) == 1
        assert service.get(memory.id) is not None
    finally:
        service.close()


def test_missing_persisted_index_rebuilds_from_sqlite_snapshot(tmp_path, monkeypatch):
    pytest.importorskip("hnswlib")
    monkeypatch.setattr(
        semantic_module,
        "create_dense_provider",
        lambda settings, *, cancel_event: _FakeProvider(settings.model_revision),
    )
    path = tmp_path / "rebuild.sqlite3"
    settings = _settings(tmp_path)
    first = MemoryService(Database(path), settings=settings)
    first.create("用户喜欢茶")
    assert first.wait_for_semantic_idle(5)
    row = first._database.connection.execute(
        "SELECT index_path FROM memory_ann_generations WHERE active = 1"
    ).fetchone()
    assert row is not None
    index_path = Path(row[0])
    first.close()
    index_path.unlink()

    restored = MemoryService(Database(path), settings=settings)
    try:
        assert restored.wait_for_semantic_idle(5)
        assert restored.status()["semantic_index"] == "ready"
        assert restored.status()["semantic_index_size"] == 1
    finally:
        restored.close()


def test_async_context_build_runs_off_event_loop(tmp_path, monkeypatch):
    service = MemoryService(Database(tmp_path / "async.sqlite3"))
    try:
        service.create("用户喜欢茶", priority=8)
        started = time.monotonic()
        result = asyncio.run(service.abuild_context_prompt("茶"))
        elapsed = time.monotonic() - started
        assert "茶" in result
        assert elapsed < 2.0
    finally:
        service.close()


def test_async_context_timeout_and_cancel_do_not_wait_for_noncooperative_provider(
    tmp_path, monkeypatch
):
    pytest.importorskip("hnswlib")
    provider = _BlockingProvider()
    monkeypatch.setattr(
        semantic_module,
        "create_dense_provider",
        lambda settings, *, cancel_event: provider,
    )
    settings = _settings(tmp_path)
    settings["semantic"]["timeout_seconds"] = 0.1
    database = Database(tmp_path / "blocking.sqlite3")
    service = MemoryService(database, settings=settings)
    try:
        assert service.wait_for_semantic_idle(5)
        service.create("用户喜欢茶")
        assert service.wait_for_semantic_idle(5)
        provider.block_documents = True
        service.create("用户喜欢猫")
        assert provider.document_started.wait(1.0)

        started = time.monotonic()
        result = asyncio.run(service.abuild_context_prompt("茶"))
        assert time.monotonic() - started < 0.8
        assert "茶" in result

        provider.release.set()
        assert service.wait_for_semantic_idle(5)
        provider.block_documents = False
        provider.block_query = True
        provider.release.clear()
        provider.query_started.clear()
        task_started = threading.Event()

        async def cancel_query() -> None:
            task = asyncio.create_task(service.abuild_context_prompt("茶"))
            for _ in range(100):
                if provider.query_started.is_set():
                    task_started.set()
                    break
                await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(asyncio.wait_for(cancel_query(), timeout=0.8))
        assert task_started.is_set()
        service.close()
        # 关闭释放 gate 后，迟到线程只能结束；它没有 mark_recalled 写入路径。
        provider.release.set()
        time.sleep(0.05)
    finally:
        if not provider.closed:
            provider.release.set()
            service.close()


def test_async_context_timeout_propagates_cancel_event_to_provider(tmp_path, monkeypatch):
    pytest.importorskip("hnswlib")
    provider = _CancelAwareProvider()
    monkeypatch.setattr(
        semantic_module,
        "create_dense_provider",
        lambda settings, *, cancel_event: provider,
    )
    settings = _settings(tmp_path)
    settings["semantic"]["timeout_seconds"] = 0.05
    service = MemoryService(Database(tmp_path / "cancel-aware.sqlite3"), settings=settings)
    try:
        assert service.wait_for_semantic_idle(5)
        service.create("用户喜欢茶")
        assert service.wait_for_semantic_idle(5)
        started = time.monotonic()
        result = asyncio.run(service.abuild_context_prompt("茶"))
        assert time.monotonic() - started < 0.8
        assert "茶" in result
        assert provider.cancel_seen.wait(0.5)
    finally:
        service.close()
