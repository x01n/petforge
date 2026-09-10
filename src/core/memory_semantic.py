"""离线稠密语义嵌入进程与持久 HNSW 索引边界。"""

from __future__ import annotations

import json
import math
import os
import selectors
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from services.processes import process_group_spawn_kwargs, terminate_process_sync

MAX_ANN_INDEX_BYTES = 512 * 1024 * 1024
MAX_PROVIDER_RESPONSE_BYTES = 64 * 1024 * 1024
_PROVIDER_ERROR_CODES = frozenset(
    {
        "dependency_unavailable",
        "model_path_missing",
        "model_load_failed",
        "embedding_dimension_mismatch",
        "embedding_failed",
        "embedding_batch_invalid",
    }
)


class SemanticMemoryError(RuntimeError):
    """只向上层暴露稳定错误码，避免诊断泄漏本地路径或模型内容。"""

    def __init__(self, code: str) -> None:
        normalized = str(code or "semantic_memory_failed").strip() or "semantic_memory_failed"
        self.code = normalized
        super().__init__(normalized)


@dataclass(frozen=True, slots=True)
class SemanticModelSpec:
    """会影响向量兼容性的完整模型身份。"""

    provider: str
    package_version: str
    model_id: str
    model_revision: str
    dimension: int


def normalize_dense_vector(value: object, dimension: int) -> tuple[float, ...]:
    """校验并归一化一个有限维 float32 兼容向量。"""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise SemanticMemoryError("invalid_embedding_shape")
    if len(value) != int(dimension):
        raise SemanticMemoryError("embedding_dimension_mismatch")
    normalized: list[float] = []
    for item in value:
        try:
            parsed = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SemanticMemoryError("invalid_embedding_value") from exc
        if not math.isfinite(parsed):
            raise SemanticMemoryError("invalid_embedding_value")
        normalized.append(parsed)
    norm = math.sqrt(sum(item * item for item in normalized))
    if not math.isfinite(norm) or norm <= 0:
        raise SemanticMemoryError("invalid_embedding_norm")
    return tuple(item / norm for item in normalized)


class DenseEmbeddingProvider(Protocol):
    """语义模型运行时需要满足的最小同步接口。"""

    @property
    def spec(self) -> SemanticModelSpec: ...

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        cancel_event: threading.Event | None = None,
    ) -> tuple[tuple[float, ...], ...]: ...

    def embed_query(
        self,
        text: str,
        *,
        cancel_event: threading.Event | None = None,
    ) -> tuple[float, ...]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SentenceTransformerProcessSettings:
    """传给隔离模型进程的已校验配置。"""

    model_path: Path
    model_id: str
    model_revision: str
    dimension: int
    batch_size: int
    timeout_seconds: float
    startup_timeout_seconds: float
    shutdown_timeout_seconds: float
    max_text_chars: int
    max_text_bytes: int
    device: str = "cpu"
    backend: str = "torch"


class SentenceTransformerProcess:
    """在可终止子进程中加载本地 SentenceTransformer。

    主进程不导入 PyTorch，也不在 Qt/asyncio 线程执行推理。每次请求带序号，
    超时会终止整个模型进程，因此迟到结果不可能进入后续请求。
    """

    def __init__(self, settings: SentenceTransformerProcessSettings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._request_id = 0
        self._closed = False
        self._close_event = threading.Event()
        self._spec: SemanticModelSpec | None = None
        self._read_buffer = bytearray()

    @property
    def spec(self) -> SemanticModelSpec:
        spec = self._spec
        if spec is None:
            raise SemanticMemoryError("provider_not_started")
        return spec

    def start(self, *, cancel_event: threading.Event | None = None) -> SemanticModelSpec:
        with self._lock:
            if self._closed or self._close_event.is_set():
                raise SemanticMemoryError("provider_closed")
            if self._spec is not None:
                return self._spec
            model_path = self._settings.model_path.expanduser().resolve()
            if not model_path.is_dir():
                raise SemanticMemoryError("model_path_missing")
            process_kwargs: dict[str, object] = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.DEVNULL,
                "env": semantic_worker_environment(),
                "close_fds": True,
            }
            process_kwargs.update(process_group_spawn_kwargs())
            process = subprocess.Popen(
                [sys.executable, "-m", "core.memory_semantic_worker"],
                **process_kwargs,
            )
            self._process = process
            self._read_buffer.clear()
            if os.name != "nt" and process.stdout is not None:
                try:
                    os.set_blocking(process.stdout.fileno(), False)
                except (OSError, ValueError):
                    pass
            try:
                self._send({"id": 0, "operation": "bootstrap", **self._worker_payload(model_path)})
                response = self._receive(
                    request_id=0,
                    timeout_seconds=self._settings.startup_timeout_seconds,
                    cancel_event=cancel_event,
                )
                raw_spec = response.get("spec")
                if not isinstance(raw_spec, dict):
                    raise SemanticMemoryError("provider_handshake_invalid")
                spec = SemanticModelSpec(
                    provider=str(raw_spec.get("provider") or ""),
                    package_version=str(raw_spec.get("package_version") or ""),
                    model_id=str(raw_spec.get("model_id") or ""),
                    model_revision=str(raw_spec.get("model_revision") or ""),
                    dimension=int(raw_spec.get("dimension") or 0),
                )
                if (
                    spec.provider != "sentence_transformers"
                    or spec.model_id != self._settings.model_id
                    or spec.model_revision != self._settings.model_revision
                    or spec.dimension != self._settings.dimension
                    or not spec.package_version
                ):
                    raise SemanticMemoryError("provider_handshake_mismatch")
                self._spec = spec
                return spec
            except BaseException:
                self._terminate()
                raise

    def _worker_payload(self, model_path: Path) -> dict[str, object]:
        return {
            "model_path": str(model_path),
            "model_id": self._settings.model_id,
            "model_revision": self._settings.model_revision,
            "dimension": self._settings.dimension,
            "batch_size": self._settings.batch_size,
            "device": self._settings.device,
            "backend": self._settings.backend,
        }

    def _send(self, payload: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise SemanticMemoryError("provider_unavailable")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_PROVIDER_RESPONSE_BYTES:
            raise SemanticMemoryError("embedding_request_too_large")
        try:
            process.stdin.write(encoded + b"\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise SemanticMemoryError("provider_exited") from exc

    def _validate_texts(self, texts: Sequence[str]) -> tuple[str, ...]:
        if isinstance(texts, (str, bytes, bytearray)) or not isinstance(texts, Sequence):
            raise SemanticMemoryError("embedding_batch_invalid")
        if not texts or len(texts) > self._settings.batch_size:
            raise SemanticMemoryError("embedding_batch_size_exceeded")
        result: list[str] = []
        for item in texts:
            text = str(item or "").strip()
            if not text:
                raise SemanticMemoryError("embedding_text_empty")
            if len(text) > self._settings.max_text_chars:
                raise SemanticMemoryError("embedding_text_too_large")
            if len(text.encode("utf-8")) > self._settings.max_text_bytes:
                raise SemanticMemoryError("embedding_text_too_large")
            result.append(text)
        return tuple(result)

    def _request(
        self,
        operation: str,
        texts: Sequence[str],
        *,
        cancel_event: threading.Event | None,
    ) -> tuple[tuple[float, ...], ...]:
        safe_texts = self._validate_texts(texts)
        with self._lock:
            if self._closed or self._close_event.is_set():
                raise SemanticMemoryError("provider_closed")
            if self._spec is None:
                self.start(cancel_event=cancel_event)
            process = self._process
            if process is None or process.poll() is not None:
                raise SemanticMemoryError("provider_unavailable")
            self._request_id += 1
            request_id = self._request_id
            try:
                self._send({"id": request_id, "operation": operation, "texts": safe_texts})
                response = self._receive(
                    request_id=request_id,
                    timeout_seconds=self._settings.timeout_seconds,
                    cancel_event=cancel_event,
                )
            except BaseException:
                self._terminate()
                raise
        values = response.get("vectors")
        if not isinstance(values, list) or len(values) != len(safe_texts):
            raise SemanticMemoryError("embedding_response_invalid")
        return tuple(normalize_dense_vector(vector, self._settings.dimension) for vector in values)

    def _receive(
        self,
        *,
        request_id: int,
        timeout_seconds: float,
        cancel_event: threading.Event | None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.01, float(timeout_seconds))
        process = self._process
        selector = selectors.DefaultSelector()
        if process is None or process.stdout is None:
            raise SemanticMemoryError("provider_unavailable")
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                if self._close_event.is_set() or (
                    cancel_event is not None and cancel_event.is_set()
                ):
                    raise SemanticMemoryError("cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SemanticMemoryError("timeout")
                events = selector.select(min(0.05, remaining))
                if events:
                    try:
                        if os.name == "nt":
                            line = process.stdout.readline(MAX_PROVIDER_RESPONSE_BYTES + 1)
                            if len(line) > MAX_PROVIDER_RESPONSE_BYTES:
                                raise SemanticMemoryError("provider_response_too_large")
                        else:
                            chunk = os.read(process.stdout.fileno(), 65_536)
                            if not chunk:
                                raise SemanticMemoryError("provider_exited")
                            self._read_buffer.extend(chunk)
                            if len(self._read_buffer) > MAX_PROVIDER_RESPONSE_BYTES:
                                raise SemanticMemoryError("provider_response_too_large")
                            newline = self._read_buffer.find(b"\n")
                            if newline < 0:
                                continue
                            line = bytes(self._read_buffer[:newline])
                            del self._read_buffer[: newline + 1]
                        response = json.loads(line.decode("utf-8"))
                    except (EOFError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise SemanticMemoryError("provider_exited") from exc
                    if not isinstance(response, dict) or int(response.get("id", -1)) != request_id:
                        raise SemanticMemoryError("provider_protocol_invalid")
                    if response.get("status") != "ok":
                        error_code = str(response.get("error") or "provider_failed")
                        if error_code not in _PROVIDER_ERROR_CODES:
                            error_code = "provider_failed"
                        raise SemanticMemoryError(error_code)
                    return response
                if process.poll() is not None:
                    raise SemanticMemoryError("provider_exited")
        finally:
            selector.close()

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        cancel_event: threading.Event | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        return self._request("documents", texts, cancel_event=cancel_event)

    def embed_query(
        self,
        text: str,
        *,
        cancel_event: threading.Event | None = None,
    ) -> tuple[float, ...]:
        return self._request("query", (text,), cancel_event=cancel_event)[0]

    def _terminate(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            terminate_process_sync(
                process,
                timeout=min(1.0, self._settings.shutdown_timeout_seconds),
            )
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            try:
                process.close()
            except AttributeError:
                pass
        self._spec = None
        self._read_buffer.clear()

    def close(self) -> None:
        # 先发出无锁取消信号，让并行 _receive 立即结束，再等待同一锁收敛。
        self._close_event.set()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    self._send({"id": -1, "operation": "close"})
                    self._receive(
                        request_id=-1,
                        timeout_seconds=self._settings.shutdown_timeout_seconds,
                        cancel_event=None,
                    )
                except (SemanticMemoryError, OSError):
                    pass
            self._terminate()


def semantic_worker_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """构造不含密钥、代理和任意业务变量的模型 worker 环境。"""

    source_values = dict(source if source is not None else os.environ)
    allowed = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "SYSTEMROOT",
        "PATHEXT",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "SENTENCE_TRANSFORMERS_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_HOME",
        "VIRTUAL_ENV",
    }
    result = {key: value for key, value in source_values.items() if key in allowed}
    # 仅注入当前安装包/源码的可验证根目录，不继承调用方的任意 PYTHONPATH。
    package_root = Path(__file__).resolve().parent.parent
    if package_root.is_dir():
        result["PYTHONPATH"] = str(package_root)
    result.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return result


class PersistentHnswIndex:
    """所有 HNSW 读写与序列化都受同一锁保护。"""

    def __init__(
        self,
        *,
        dimension: int,
        capacity: int,
        m: int,
        ef_construction: int,
        ef_search: int,
    ) -> None:
        try:
            import hnswlib
        except ImportError as exc:
            raise SemanticMemoryError("dependency_unavailable") from exc
        self.dimension = int(dimension)
        self.capacity = max(16, int(capacity))
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self._hnswlib = hnswlib
        self._index: Any | None = None
        self._active_ids: set[int] = set()
        self._lock = threading.RLock()

    @classmethod
    def build(
        cls,
        items: Iterable[tuple[int, Sequence[float]]],
        *,
        dimension: int,
        capacity: int,
        m: int,
        ef_construction: int,
        ef_search: int,
    ) -> PersistentHnswIndex:
        instance = cls(
            dimension=dimension,
            capacity=capacity,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        prepared = tuple(
            (int(memory_id), normalize_dense_vector(vector, dimension))
            for memory_id, vector in items
        )
        index = instance._hnswlib.Index(space="cosine", dim=instance.dimension)
        index.init_index(
            max_elements=max(instance.capacity, len(prepared) + 16),
            M=instance.m,
            ef_construction=instance.ef_construction,
            random_seed=100,
            allow_replace_deleted=True,
        )
        index.set_ef(instance.ef_search)
        index.set_num_threads(1)
        if prepared:
            index.add_items(
                [list(vector) for _, vector in prepared],
                [memory_id for memory_id, _ in prepared],
                num_threads=1,
            )
        instance._index = index
        instance._active_ids = {memory_id for memory_id, _ in prepared}
        return instance

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        active_ids: Iterable[int],
        dimension: int,
        capacity: int,
        m: int,
        ef_construction: int,
        ef_search: int,
    ) -> PersistentHnswIndex:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise SemanticMemoryError("index_file_missing")
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise SemanticMemoryError("index_file_invalid") from exc
        if size <= 0 or size > MAX_ANN_INDEX_BYTES:
            raise SemanticMemoryError("index_file_size_invalid")
        instance = cls(
            dimension=dimension,
            capacity=capacity,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        index = instance._hnswlib.Index(space="cosine", dim=instance.dimension)
        try:
            index.load_index(
                str(resolved),
                max_elements=max(instance.capacity, 16),
                allow_replace_deleted=True,
            )
            # hnswlib 的序列化契约不保证调用方设置的 ef，加载后必须显式恢复。
            index.set_ef(instance.ef_search)
            index.set_num_threads(1)
        except BaseException as exc:
            raise SemanticMemoryError("index_load_failed") from exc
        instance._index = index
        instance._active_ids = {int(item) for item in active_ids}
        return instance

    def upsert(self, memory_id: int, vector: Sequence[float]) -> None:
        identity = int(memory_id)
        normalized = normalize_dense_vector(vector, self.dimension)
        with self._lock:
            index = self._require_index()
            if identity in self._active_ids:
                # hnswlib 不允许对活动标签重复 add_items；先标记旧向量，
                # 再用相同标签写入新向量，保持增量更新语义。
                index.mark_deleted(identity)
                self._active_ids.remove(identity)
            if len(self._active_ids) >= index.max_elements:
                index.resize_index(max(index.max_elements * 2, len(self._active_ids) + 16))
            index.add_items(
                [list(normalized)],
                [identity],
                num_threads=1,
                replace_deleted=True,
            )
            self._active_ids.add(identity)

    def remove(self, memory_id: int) -> None:
        identity = int(memory_id)
        with self._lock:
            if identity not in self._active_ids:
                return
            self._require_index().mark_deleted(identity)
            self._active_ids.remove(identity)

    def search(self, vector: Sequence[float], *, limit: int) -> tuple[tuple[int, float], ...]:
        normalized = normalize_dense_vector(vector, self.dimension)
        with self._lock:
            safe_limit = min(max(0, int(limit)), len(self._active_ids))
            if safe_limit == 0:
                return ()
            labels, distances = self._require_index().knn_query(
                [list(normalized)],
                k=safe_limit,
                num_threads=1,
            )
            result: list[tuple[int, float]] = []
            for raw_label, raw_distance in zip(labels[0], distances[0], strict=True):
                memory_id = int(raw_label)
                if memory_id not in self._active_ids:
                    continue
                score = 1.0 - float(raw_distance)
                if math.isfinite(score):
                    result.append((memory_id, max(-1.0, min(1.0, score))))
            return tuple(result)

    def save_revision(self, directory: Path, revision: str) -> Path:
        resolved_directory = directory.expanduser().resolve()
        resolved_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(resolved_directory, 0o700)
        except OSError:
            pass
        safe_revision = str(revision or "")
        if len(safe_revision) != 32 or any(
            char not in "0123456789abcdef" for char in safe_revision
        ):
            raise SemanticMemoryError("index_revision_invalid")
        destination = resolved_directory / f"{safe_revision}.hnsw"
        temporary = resolved_directory / f".{safe_revision}.{uuid.uuid4().hex}.tmp"
        with self._lock:
            try:
                self._require_index().save_index(str(temporary))
                if temporary.stat().st_size <= 0 or temporary.stat().st_size > MAX_ANN_INDEX_BYTES:
                    raise SemanticMemoryError("index_file_size_invalid")
                os.chmod(temporary, 0o600)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                try:
                    directory_fd = os.open(resolved_directory, os.O_RDONLY | os.O_DIRECTORY)
                except (AttributeError, OSError):
                    directory_fd = -1
                if directory_fd >= 0:
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            finally:
                if temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
        return destination

    def _require_index(self) -> Any:
        if self._index is None:
            raise SemanticMemoryError("index_not_initialized")
        return self._index

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._active_ids)

    @property
    def active_ids(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._active_ids)
