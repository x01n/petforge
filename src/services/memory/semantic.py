"""长期记忆的后台语义嵌入、持久 ANN 与 SQLite 一致性协调。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.memory_semantic import (
    DenseEmbeddingProvider,
    PersistentHnswIndex,
    SemanticMemoryError,
    SemanticModelSpec,
    SentenceTransformerProcess,
    SentenceTransformerProcessSettings,
    normalize_dense_vector,
)
from db.database import Database

MAX_MODEL_PATH_CHARS = 4096
MAX_SEMANTIC_MUTATIONS_PER_PASS = 2_000
_GENERATION_CHARS = frozenset("0123456789abcdef")
_PUBLIC_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,95}$")
_SENSITIVE_MODEL_ID_RE = re.compile(
    r"(?:^|[-_])(sk|token|secret|password|passwd|api[-_]?key)(?:$|[-_])|://|\\",
    re.IGNORECASE,
)


def _public_model_identity(value: object) -> str:
    """返回可展示的模型标识；路径、URL 和疑似凭据只保留短指纹。"""

    normalized = str(value or "").strip()
    if not normalized:
        return ""
    if (
        _PUBLIC_MODEL_ID_RE.fullmatch(normalized)
        and not normalized.startswith(("/", ".", "~"))
        and normalized.count("/") <= 1
        and ":" not in normalized
        and not _SENSITIVE_MODEL_ID_RE.search(normalized)
    ):
        return normalized
    return f"sha256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:12]}"


@dataclass(frozen=True, slots=True)
class SemanticMemorySettings:
    """显式启用的本地语义模型与 HNSW 运行策略。"""

    enabled: bool = False
    provider: str = "sentence_transformers"
    model_path: str = ""
    model_id: str = ""
    model_revision: str = ""
    dimension: int = 384
    batch_size: int = 16
    timeout_seconds: float = 5.0
    startup_timeout_seconds: float = 60.0
    shutdown_timeout_seconds: float = 2.0
    max_text_chars: int = 10_000
    max_text_bytes: int = 40_000
    device: str = "cpu"
    backend: str = "torch"
    index_dir: str = ""
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        base_directory: str | Path | None = None,
    ) -> SemanticMemorySettings:
        """严格解析配置；默认关闭且不会加载模型或访问网络。"""

        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("memory.semantic must be a mapping")

        def boolean(name: str, default: bool) -> bool:
            raw = value.get(name, default)
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)) and raw in {0, 1}:
                return bool(raw)
            normalized = str(raw or "").strip().lower()
            if normalized in {"true", "yes", "on", "1"}:
                return True
            if normalized in {"false", "no", "off", "0"}:
                return False
            raise ValueError(f"memory.semantic.{name} must be a boolean")

        def integer(name: str, default: int, low: int, high: int) -> int:
            raw = value.get(name, default)
            if isinstance(raw, bool):
                raise ValueError(f"memory.semantic.{name} must be an integer")
            try:
                parsed = int(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"memory.semantic.{name} must be an integer") from exc
            if parsed != raw or parsed < low or parsed > high:
                raise ValueError(f"memory.semantic.{name} is outside the allowed range")
            return parsed

        def number(name: str, default: float, low: float, high: float) -> float:
            raw = value.get(name, default)
            if isinstance(raw, bool):
                raise ValueError(f"memory.semantic.{name} must be a number")
            try:
                parsed = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"memory.semantic.{name} must be a number") from exc
            if not math.isfinite(parsed) or parsed < low or parsed > high:
                raise ValueError(f"memory.semantic.{name} is outside the allowed range")
            return parsed

        def text(name: str, default: str = "", *, required: bool = False) -> str:
            raw = value.get(name, default)
            if raw is None:
                raw = ""
            if not isinstance(raw, str):
                raise ValueError(f"memory.semantic.{name} must be a string")
            parsed = raw.strip()
            if any(character in parsed for character in "\x00\r\n"):
                raise ValueError(f"memory.semantic.{name} is invalid")
            if len(parsed) > MAX_MODEL_PATH_CHARS:
                raise ValueError(f"memory.semantic.{name} is too long")
            if required and not parsed:
                raise ValueError(f"memory.semantic.{name} is required when enabled")
            return parsed

        enabled = boolean("enabled", False)
        provider = text("provider", "sentence_transformers", required=enabled)
        if provider != "sentence_transformers":
            raise ValueError("memory.semantic.provider is unsupported")
        model_path = text("model_path")
        index_dir = text("index_dir")
        if base_directory is not None:
            base = Path(base_directory).expanduser().resolve()
            if model_path:
                path = Path(model_path).expanduser()
                model_path = str(path.resolve() if path.is_absolute() else (base / path).resolve())
            if index_dir:
                path = Path(index_dir).expanduser()
                index_dir = str(path.resolve() if path.is_absolute() else (base / path).resolve())
        device = text("device", "cpu", required=True)
        if device not in {"cpu", "cuda", "mps"} and not (
            device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise ValueError("memory.semantic.device is unsupported")
        backend = text("backend", "torch", required=True)
        if backend not in {"torch", "onnx", "openvino"}:
            raise ValueError("memory.semantic.backend is unsupported")
        return cls(
            enabled=enabled,
            provider=provider,
            model_path=model_path,
            model_id=text("model_id"),
            model_revision=text("model_revision"),
            dimension=integer("dimension", 384, 8, 4_096),
            batch_size=integer("batch_size", 16, 1, 256),
            timeout_seconds=number("timeout_seconds", 5.0, 0.05, 300.0),
            startup_timeout_seconds=number("startup_timeout_seconds", 60.0, 1.0, 600.0),
            shutdown_timeout_seconds=number("shutdown_timeout_seconds", 2.0, 0.1, 30.0),
            max_text_chars=integer("max_text_chars", 10_000, 32, 50_000),
            max_text_bytes=integer("max_text_bytes", 40_000, 128, 200_000),
            device=device,
            backend=backend,
            index_dir=index_dir,
            hnsw_m=integer("hnsw_m", 16, 2, 64),
            hnsw_ef_construction=integer("hnsw_ef_construction", 200, 8, 2_000),
            hnsw_ef_search=integer("hnsw_ef_search", 64, 2, 2_000),
        )

    def fingerprint(self, spec: SemanticModelSpec) -> str:
        """生成必须触发索引重建的稳定身份摘要。"""

        payload = {
            "provider": spec.provider,
            "package_version": spec.package_version,
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
            "dimension": spec.dimension,
            "backend": self.backend,
            "device": self.device,
            "model_path": self.model_path,
            "index_dir": self.index_dir,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "format": 1,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def create_dense_provider(
    settings: SemanticMemorySettings,
    *,
    cancel_event: threading.Event,
) -> DenseEmbeddingProvider:
    """创建并启动默认的离线模型进程。"""

    if not settings.model_path or not settings.model_id or not settings.model_revision:
        raise SemanticMemoryError("model_config_missing")

    provider = SentenceTransformerProcess(
        SentenceTransformerProcessSettings(
            model_path=Path(settings.model_path),
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            dimension=settings.dimension,
            batch_size=settings.batch_size,
            timeout_seconds=settings.timeout_seconds,
            startup_timeout_seconds=settings.startup_timeout_seconds,
            shutdown_timeout_seconds=settings.shutdown_timeout_seconds,
            max_text_chars=settings.max_text_chars,
            max_text_bytes=settings.max_text_bytes,
            device=settings.device,
            backend=settings.backend,
        )
    )
    provider.start(cancel_event=cancel_event)
    return provider


def _content_digest(content: object) -> str:
    return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()


def _encode_vector(vector: Sequence[float], dimension: int) -> bytes:
    normalized = normalize_dense_vector(vector, dimension)
    values = array("f", normalized)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _decode_vector(payload: object, dimension: int) -> tuple[float, ...]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise SemanticMemoryError("stored_embedding_invalid")
    raw = bytes(payload)
    if len(raw) != int(dimension) * 4:
        raise SemanticMemoryError("stored_embedding_invalid")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return normalize_dense_vector(tuple(values), dimension)


def _valid_identity(value: object) -> str:
    normalized = str(value or "")
    if len(normalized) != 32 or any(char not in _GENERATION_CHARS for char in normalized):
        raise SemanticMemoryError("index_identity_invalid")
    return normalized


@dataclass(slots=True)
class _ActiveSemanticIndex:
    generation: str
    index_revision: str
    index_path: Path
    applied_sequence: int
    provider: DenseEmbeddingProvider
    index: PersistentHnswIndex
    spec: SemanticModelSpec
    fingerprint: str


class SemanticMemoryController:
    """串行后台构建，并以 SQLite active 指针原子切换索引代次。"""

    def __init__(
        self,
        database: Database,
        settings: SemanticMemorySettings,
        *,
        capacity: int,
    ) -> None:
        if not isinstance(database, Database):
            raise TypeError("database must be a Database")
        self._database = database
        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._desired = settings
        self._capacity = max(16, int(capacity))
        self._serial = 1 if settings.enabled else 0
        self._handled_serial = 0
        self._sync_requested = False
        self._working = False
        self._thread: threading.Thread | None = None
        self._active: _ActiveSemanticIndex | None = None
        self._building_provider: DenseEmbeddingProvider | None = None
        self._phase = "initializing" if settings.enabled else "disabled"
        self._reason = ""
        self._closed = False
        if settings.enabled:
            self._ensure_thread()

    def _ensure_thread(self) -> None:
        with self._condition:
            thread = self._thread
            if thread is not None and thread.is_alive():
                self._condition.notify_all()
                return
            if self._closed:
                return
            thread = threading.Thread(
                target=self._worker_main,
                daemon=False,
                name="meapet-semantic-index",
            )
            self._thread = thread
            thread.start()

    def configure(self, settings: SemanticMemorySettings, *, capacity: int) -> None:
        """提交新配置；模型加载与索引重建始终留在后台线程。"""

        with self._condition:
            if self._closed:
                return
            changed = settings != self._desired or int(capacity) != self._capacity
            self._desired = settings
            self._capacity = max(16, int(capacity))
            if changed:
                self._serial += 1
                self._sync_requested = False
                self._phase = "rebuilding" if settings.enabled else "disabled"
                self._reason = ""
            self._condition.notify_all()
        if settings.enabled or self._active is not None:
            self._ensure_thread()
        elif not settings.enabled:
            self._clear_journal()

    def notify_mutation(self) -> None:
        """通知后台消费由 SQLite trigger 记录的增删改日志。"""

        with self._condition:
            if self._closed or not self._desired.enabled:
                if not self._closed:
                    self._clear_journal()
                return
            self._sync_requested = True
            self._condition.notify_all()
        self._ensure_thread()

    def _clear_journal(self) -> None:
        try:
            with self._database.transaction() as connection:
                connection.execute("DELETE FROM memory_ann_journal")
        except sqlite3.Error:
            pass

    def _worker_main(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._stop_event.is_set()
                        or self._handled_serial != self._serial
                        or self._sync_requested
                    )
                )
                if self._stop_event.is_set():
                    break
                serial = self._serial
                settings = self._desired
                rebuild = self._handled_serial != serial
                sync = self._sync_requested and not rebuild
                if rebuild:
                    self._handled_serial = serial
                if sync:
                    self._sync_requested = False
                self._working = True
            try:
                if not settings.enabled:
                    self._deactivate()
                elif rebuild:
                    self._activate(serial, settings)
                elif sync:
                    self._sync_active(serial, settings)
            finally:
                with self._condition:
                    self._working = False
                    self._condition.notify_all()

    def _activate(self, serial: int, settings: SemanticMemorySettings) -> None:
        provider: DenseEmbeddingProvider | None = None
        built_artifact: _ActiveSemanticIndex | None = None
        try:
            with self._condition:
                self._phase = "rebuilding" if self._active is not None else "initializing"
                self._reason = ""
            provider = create_dense_provider(settings, cancel_event=self._stop_event)
            with self._condition:
                self._building_provider = provider
            fingerprint = settings.fingerprint(provider.spec)
            try:
                loaded = self._load_persisted(provider, settings, fingerprint)
            except SemanticMemoryError as exc:
                if exc.code not in {
                    "index_file_missing",
                    "index_file_size_invalid",
                    "index_file_invalid",
                    "index_load_failed",
                    "index_identity_invalid",
                    "stored_embedding_invalid",
                }:
                    raise
                # 持久文件损坏/被清理时在后台从 SQLite 快照重建，旧活动代次
                # 仍保留到新文件和事务都成功为止。
                loaded = None
            vectors: dict[int, tuple[str, tuple[float, ...]]] | None = None
            if loaded is not None:
                next_active = loaded
            else:
                next_active, vectors = self._rebuild(provider, settings, fingerprint)
                built_artifact = next_active
            if self._stop_event.is_set():
                raise SemanticMemoryError("cancelled")
            with self._condition:
                if serial != self._serial or settings != self._desired:
                    raise SemanticMemoryError("superseded")
                if vectors is not None:
                    # DB active 指针提交与内存 active 替换位于同一 condition
                    # 临界区；配置 supersede 时事务尚未开始，旧代次完整保留。
                    self._commit_rebuild(next_active, vectors, settings)
                previous = self._active
                self._active = next_active
                self._phase = "ready"
                self._reason = ""
                self._building_provider = None
                self._sync_requested = self._pending_sequence_count() > 0
                self._condition.notify_all()
            provider = None
            if previous is not None:
                previous.provider.close()
                if previous.index_path != next_active.index_path:
                    self._remove_index_path(previous.index_path)
            self._cleanup_inactive_generations(next_active.generation, settings)
        except SemanticMemoryError as exc:
            self._record_failure(exc.code)
        except (OSError, RuntimeError, sqlite3.Error, ValueError):
            self._record_failure("semantic_initialization_failed")
        finally:
            with self._condition:
                self._building_provider = None
            if provider is not None:
                provider.close()
            if built_artifact is not None:
                with self._condition:
                    active = self._active
                if active is not built_artifact:
                    self._remove_index_path(built_artifact.index_path)

    def _load_persisted(
        self,
        provider: DenseEmbeddingProvider,
        settings: SemanticMemorySettings,
        fingerprint: str,
    ) -> _ActiveSemanticIndex | None:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT * FROM memory_ann_generations WHERE active = 1"
            ).fetchone()
            if row is None or str(row["settings_fingerprint"]) != fingerprint:
                return None
            generation = _valid_identity(row["generation"])
            index_revision = _valid_identity(row["index_revision"])
            vector_rows = self._database.connection.execute(
                "SELECT memory_id, vector, tombstone FROM memory_semantic_vectors "
                "WHERE generation = ? ORDER BY memory_id",
                (generation,),
            ).fetchall()
        active_ids = tuple(
            int(item["memory_id"]) for item in vector_rows if int(item["tombstone"] or 0) == 0
        )
        if len(active_ids) != int(row["item_count"] or 0):
            return None
        raw_path = str(row["index_path"] or "").strip()
        path = (
            Path(raw_path).expanduser().resolve()
            if raw_path
            else self._index_directory(settings) / f"{index_revision}.hnsw"
        )
        if path.name != f"{index_revision}.hnsw":
            return None
        if path.parent != self._index_directory(settings).expanduser().resolve():
            return None
        index = PersistentHnswIndex.load(
            path,
            active_ids=active_ids,
            dimension=settings.dimension,
            capacity=self._capacity,
            m=settings.hnsw_m,
            ef_construction=settings.hnsw_ef_construction,
            ef_search=settings.hnsw_ef_search,
        )
        return _ActiveSemanticIndex(
            generation=generation,
            index_revision=index_revision,
            index_path=path,
            applied_sequence=int(row["applied_sequence"] or 0),
            provider=provider,
            index=index,
            spec=provider.spec,
            fingerprint=fingerprint,
        )

    def _rebuild(
        self,
        provider: DenseEmbeddingProvider,
        settings: SemanticMemorySettings,
        fingerprint: str,
    ) -> tuple[_ActiveSemanticIndex, dict[int, tuple[str, tuple[float, ...]]]]:
        with self._database._lock:
            rows = self._database.connection.execute(
                "SELECT id, content FROM memories ORDER BY id"
            ).fetchall()
            sequence_row = self._database.connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM memory_ann_journal"
            ).fetchone()
        applied_sequence = int(sequence_row[0] or 0) if sequence_row else 0
        vectors: dict[int, tuple[str, tuple[float, ...]]] = {}
        for offset in range(0, len(rows), settings.batch_size):
            if self._stop_event.is_set():
                raise SemanticMemoryError("cancelled")
            batch = rows[offset : offset + settings.batch_size]
            embedded = provider.embed_documents(
                tuple(str(row["content"]) for row in batch),
                cancel_event=self._stop_event,
            )
            for row, vector in zip(batch, embedded, strict=True):
                content = str(row["content"])
                vectors[int(row["id"])] = (_content_digest(content), vector)
        index = PersistentHnswIndex.build(
            ((memory_id, value[1]) for memory_id, value in vectors.items()),
            dimension=settings.dimension,
            capacity=self._capacity,
            m=settings.hnsw_m,
            ef_construction=settings.hnsw_ef_construction,
            ef_search=settings.hnsw_ef_search,
        )
        generation = uuid.uuid4().hex
        index_revision = uuid.uuid4().hex
        index.save_revision(self._index_directory(settings), index_revision)
        return _ActiveSemanticIndex(
            generation=generation,
            index_revision=index_revision,
            index_path=self._index_directory(settings) / f"{index_revision}.hnsw",
            applied_sequence=applied_sequence,
            provider=provider,
            index=index,
            spec=provider.spec,
            fingerprint=fingerprint,
        ), vectors

    def _commit_rebuild(
        self,
        active: _ActiveSemanticIndex,
        vectors: Mapping[int, tuple[str, tuple[float, ...]]],
        settings: SemanticMemorySettings,
    ) -> None:
        """在调用方 condition 临界区内提交 active 指针和向量快照。"""

        now = time.time()
        with self._database.transaction() as connection:
            connection.execute("UPDATE memory_ann_generations SET active = 0 WHERE active = 1")
            connection.execute(
                """
                INSERT INTO memory_ann_generations (
                    generation, settings_fingerprint, provider, package_version,
                    model_id, model_revision, dimension, index_revision, index_path,
                    item_count, applied_sequence, created_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    active.generation,
                    active.fingerprint,
                    active.spec.provider,
                    active.spec.package_version,
                    active.spec.model_id,
                    active.spec.model_revision,
                    active.spec.dimension,
                    active.index_revision,
                    str(active.index_path),
                    len(vectors),
                    active.applied_sequence,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO memory_semantic_vectors (
                    generation, memory_id, content_digest, vector, dimension,
                    tombstone, updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    (
                        active.generation,
                        memory_id,
                        digest,
                        _encode_vector(vector, settings.dimension),
                        settings.dimension,
                        now,
                    )
                    for memory_id, (digest, vector) in vectors.items()
                ),
            )
            connection.execute(
                "DELETE FROM memory_ann_journal WHERE sequence <= ?", (active.applied_sequence,)
            )

    def _sync_active(self, serial: int, settings: SemanticMemorySettings) -> None:
        with self._condition:
            active = self._active
        if active is None:
            with self._condition:
                self._serial += 1
                self._condition.notify_all()
            return
        try:
            changes = self._read_changes(active.applied_sequence)
            if not changes:
                return
            latest: dict[int, tuple[int, str]] = {}
            for sequence, memory_id, operation in changes:
                latest[memory_id] = (sequence, operation)
            current = self._read_current_memories(tuple(latest))
            upserts: list[tuple[int, str]] = []
            removals: list[int] = []
            for memory_id, (_, operation) in latest.items():
                content = current.get(memory_id)
                if operation == "delete" or content is None:
                    removals.append(memory_id)
                else:
                    upserts.append((memory_id, content))
            embedded: dict[int, tuple[str, tuple[float, ...]]] = {}
            for offset in range(0, len(upserts), settings.batch_size):
                batch = upserts[offset : offset + settings.batch_size]
                vectors = active.provider.embed_documents(
                    tuple(content for _, content in batch),
                    cancel_event=self._stop_event,
                )
                for (memory_id, content), vector in zip(batch, vectors, strict=True):
                    embedded[memory_id] = (_content_digest(content), vector)
            vector_rows = self._read_active_vectors(active.generation, settings.dimension)
            clone = PersistentHnswIndex.load(
                active.index_path,
                active_ids=(memory_id for memory_id, _, _ in vector_rows),
                dimension=settings.dimension,
                capacity=self._capacity,
                m=settings.hnsw_m,
                ef_construction=settings.hnsw_ef_construction,
                ef_search=settings.hnsw_ef_search,
            )
            for memory_id in removals:
                clone.remove(memory_id)
            for memory_id, (_, vector) in embedded.items():
                clone.upsert(memory_id, vector)
            index_revision = uuid.uuid4().hex
            index_path = clone.save_revision(self._index_directory(settings), index_revision)
            max_sequence = max(sequence for sequence, _, _ in changes)
            now = time.time()
            with self._condition:
                if serial != self._serial or active is not self._active:
                    self._remove_index_path(index_path)
                    return
                try:
                    # 校验 serial 后，在同一 condition 临界区提交 SQLite 元数据
                    # 并替换内存索引；事务失败时旧代次和旧文件仍是活动状态。
                    with self._database.transaction() as connection:
                        connection.executemany(
                            """
                            INSERT INTO memory_semantic_vectors (
                                generation, memory_id, content_digest, vector, dimension,
                                tombstone, updated_at
                            ) VALUES (?, ?, '', X'', ?, 1, ?)
                            ON CONFLICT(generation, memory_id) DO UPDATE SET
                                content_digest = '', vector = X'', dimension = excluded.dimension,
                                tombstone = 1, updated_at = excluded.updated_at
                            """,
                            (
                                (active.generation, memory_id, settings.dimension, now)
                                for memory_id in removals
                            ),
                        )
                        connection.executemany(
                            """
                            INSERT INTO memory_semantic_vectors (
                                generation, memory_id, content_digest, vector, dimension,
                                tombstone, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 0, ?)
                            ON CONFLICT(generation, memory_id) DO UPDATE SET
                                content_digest = excluded.content_digest,
                                vector = excluded.vector,
                                dimension = excluded.dimension,
                                tombstone = 0,
                                updated_at = excluded.updated_at
                            """,
                            (
                                (
                                    active.generation,
                                    memory_id,
                                    digest,
                                    _encode_vector(vector, settings.dimension),
                                    settings.dimension,
                                    now,
                                )
                                for memory_id, (digest, vector) in embedded.items()
                            ),
                        )
                        metadata_cursor = connection.execute(
                            """
                            UPDATE memory_ann_generations
                            SET index_revision = ?, index_path = ?, item_count = ?,
                                applied_sequence = ?
                            WHERE generation = ? AND active = 1
                            """,
                            (
                                index_revision,
                                str(index_path),
                                clone.size,
                                max_sequence,
                                active.generation,
                            ),
                        )
                        if metadata_cursor.rowcount != 1:
                            raise SemanticMemoryError("active_generation_changed")
                        connection.execute(
                            "DELETE FROM memory_ann_journal WHERE sequence <= ?", (max_sequence,)
                        )
                except BaseException:
                    self._remove_index_path(index_path)
                    raise
                old_path = active.index_path
                active.index_revision = index_revision
                active.index_path = index_path
                active.applied_sequence = max_sequence
                active.index = clone
                self._phase = "ready"
                self._reason = ""
                self._sync_requested = self._pending_sequence_count() > 0
                self._condition.notify_all()
            if old_path != index_path:
                self._remove_index_path(old_path)
        except SemanticMemoryError as exc:
            self._schedule_rebuild(exc.code)
        except (OSError, RuntimeError, sqlite3.Error, ValueError):
            self._schedule_rebuild("semantic_sync_failed")

    def _read_changes(self, applied_sequence: int) -> tuple[tuple[int, int, str], ...]:
        with self._database._lock:
            rows = self._database.connection.execute(
                """
                SELECT sequence, memory_id, operation
                FROM memory_ann_journal
                WHERE sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (int(applied_sequence), MAX_SEMANTIC_MUTATIONS_PER_PASS),
            ).fetchall()
        return tuple(
            (int(row["sequence"]), int(row["memory_id"]), str(row["operation"])) for row in rows
        )

    def _read_current_memories(self, memory_ids: Sequence[int]) -> dict[int, str]:
        result: dict[int, str] = {}
        for offset in range(0, len(memory_ids), 400):
            chunk = tuple(memory_ids[offset : offset + 400])
            if not chunk:
                continue
            with self._database._lock:
                rows = self._database.connection.execute(
                    f"SELECT id, content FROM memories WHERE id IN "
                    f"({','.join('?' for _ in chunk)})",
                    chunk,
                ).fetchall()
            result.update((int(row["id"]), str(row["content"])) for row in rows)
        return result

    def _read_active_vectors(
        self,
        generation: str,
        dimension: int,
    ) -> tuple[tuple[int, str, tuple[float, ...]], ...]:
        with self._database._lock:
            rows = self._database.connection.execute(
                """
                SELECT memory_id, content_digest, vector
                FROM memory_semantic_vectors
                WHERE generation = ? AND tombstone = 0
                ORDER BY memory_id
                """,
                (generation,),
            ).fetchall()
        return tuple(
            (
                int(row["memory_id"]),
                str(row["content_digest"]),
                _decode_vector(row["vector"], dimension),
            )
            for row in rows
        )

    def search(
        self,
        query: str,
        *,
        limit: int,
        cancel_event: threading.Event | None = None,
    ) -> tuple[tuple[int, float], ...]:
        """同步查询入口；调用方须放入工作线程，失败时返回空结果。"""

        with self._condition:
            active = self._active
            if (
                active is None
                or self._closed
                or not self._desired.enabled
                or (cancel_event is not None and cancel_event.is_set())
            ):
                return ()
        try:
            # 同时传入 controller 关闭事件和本次请求取消事件；真实 provider
            # 会据此终止当前模型进程，替身即使忽略取消也只会留在 daemon 线程。
            request_cancel = cancel_event
            if request_cancel is None:
                request_cancel = self._stop_event
            vector = active.provider.embed_query(query, cancel_event=request_cancel)
            with self._condition:
                if (
                    active is not self._active
                    or self._closed
                    or (cancel_event is not None and cancel_event.is_set())
                ):
                    return ()
            result = active.index.search(vector, limit=limit)
        except SemanticMemoryError as exc:
            if exc.code == "cancelled" and cancel_event is not None and cancel_event.is_set():
                # 单次会话取消不应把正常索引标记为失败；真实 provider 已被
                # 终止时下一次查询会按 provider_unavailable 触发重建。
                return ()
            self._schedule_rebuild(exc.code)
            return ()
        except (OSError, RuntimeError, ValueError):
            if cancel_event is not None and cancel_event.is_set():
                return ()
            self._schedule_rebuild("semantic_query_failed")
            return ()
        return result

    def _schedule_rebuild(self, reason: str) -> None:
        with self._condition:
            if self._closed or not self._desired.enabled:
                return
            self._serial += 1
            self._phase = "degraded"
            self._reason = str(reason or "semantic_memory_failed")
            self._sync_requested = False
            self._condition.notify_all()
        self._ensure_thread()

    def _record_failure(self, reason: str) -> None:
        with self._condition:
            if self._stop_event.is_set() or reason in {"cancelled", "superseded"}:
                return
            self._phase = "degraded"
            self._reason = str(reason or "semantic_memory_failed")
            self._condition.notify_all()

    def _deactivate(self) -> None:
        with self._condition:
            active, self._active = self._active, None
            self._phase = "disabled"
            self._reason = ""
            self._sync_requested = False
            self._condition.notify_all()
        if active is not None:
            active.provider.close()
        self._clear_journal()

    def _index_directory(self, settings: SemanticMemorySettings) -> Path:
        if settings.index_dir:
            return Path(settings.index_dir).expanduser().resolve()
        database_path = str(self._database.path)
        if database_path == ":memory:":
            raise SemanticMemoryError("persistent_index_requires_file_database")
        path = Path(database_path).expanduser().resolve()
        return path.with_name(f"{path.name}.semantic-index")

    def _remove_index_path(self, path: Path) -> None:
        try:
            # 只按词法路径删除生成的普通文件；绝不 resolve 文件本身，
            # 否则恶意 symlink 会把删除操作转移到索引目录之外的目标。
            candidate = Path(path).expanduser()
            if (
                not candidate.is_absolute()
                or candidate.suffix != ".hnsw"
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                return
            candidate.unlink()
        except OSError:
            pass

    def _cleanup_inactive_generations(
        self,
        active_generation: str,
        settings: SemanticMemorySettings,
    ) -> None:
        try:
            with self._database.transaction() as connection:
                rows = connection.execute(
                    "SELECT generation, index_revision, index_path FROM memory_ann_generations "
                    "WHERE generation != ?",
                    (active_generation,),
                ).fetchall()
                connection.execute(
                    "DELETE FROM memory_semantic_vectors WHERE generation != ?",
                    (active_generation,),
                )
                connection.execute(
                    "DELETE FROM memory_ann_generations WHERE generation != ?",
                    (active_generation,),
                )
            for row in rows:
                raw_path = str(row["index_path"] or "").strip()
                try:
                    identity = _valid_identity(row["index_revision"])
                    base_directory = self._index_directory(settings).expanduser().resolve()
                    expected_name = f"{identity}.hnsw"
                    path = (
                        Path(os.path.abspath(os.path.expanduser(raw_path)))
                        if raw_path
                        else base_directory / expected_name
                    )
                except (OSError, SemanticMemoryError):
                    continue
                # SQLite 可能来自旧版本、备份或不可信导入；只允许删除当前
                # 配置目录中与 revision 精确匹配的普通索引文件，绝不信任
                # 持久化的任意绝对路径。
                if (
                    path.parent.resolve() != base_directory
                    or path.name != expected_name
                    or path.is_symlink()
                ):
                    continue
                self._remove_index_path(path)
        except (OSError, sqlite3.Error):
            pass

    def _pending_sequence_count(self) -> int:
        try:
            with self._database._lock:
                row = self._database.connection.execute(
                    "SELECT COUNT(*) FROM memory_ann_journal"
                ).fetchone()
            return int(row[0] or 0) if row else 0
        except sqlite3.Error:
            return 0

    def status(self) -> dict[str, object]:
        """返回不含本地模型路径和向量内容的即时诊断。"""

        with self._condition:
            active = self._active
            desired = self._desired
            phase = self._phase
            reason = self._reason
            working = self._working
        return {
            "enabled": desired.enabled,
            "status": phase,
            "degraded_reason": reason,
            "provider": desired.provider,
            "model_id": _public_model_identity(desired.model_id),
            "model_revision": _public_model_identity(desired.model_revision),
            "dimension": desired.dimension,
            "batch_size": desired.batch_size,
            "timeout_seconds": desired.timeout_seconds,
            "index_size": active.index.size if active is not None else 0,
            "active_generation": active.generation if active is not None else "",
            "active_model_id": (
                _public_model_identity(active.spec.model_id) if active is not None else ""
            ),
            "active_model_revision": (
                _public_model_identity(active.spec.model_revision) if active is not None else ""
            ),
            "active_package_version": active.spec.package_version if active is not None else "",
            "pending_mutations": self._pending_sequence_count(),
            "working": working,
        }

    def wait_for_idle(self, timeout_seconds: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        with self._condition:
            while self._working or self._handled_serial != self._serial or self._sync_requested:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            self._condition.notify_all()
            thread = self._thread
            active = self._active
            building = self._building_provider
            timeout = self._desired.shutdown_timeout_seconds
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if building is not None and building is not active:
            building.close()
        if active is not None:
            active.provider.close()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        with self._condition:
            unsettled = thread is not None and thread.is_alive()
            self._active = None
            self._phase = "closed"
            self._reason = "shutdown_timeout" if unsettled else ""
            self._condition.notify_all()
