"""用于长期记忆检索的确定性稀疏嵌入。"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Iterable, Sequence

VECTOR_DIM = 2048
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[\u3040-\u30ff]+")


def token_hash(token: str) -> int:
    """将 token 稳定映射到固定维度。"""

    value = 0
    for character in token:
        value = (value * 31 + ord(character)) & 0xFFFFFFFF
    return value % VECTOR_DIM


def compute_embedding(text: str | None) -> list[tuple[int, float]]:
    """生成词/字符混合的归一化稀疏嵌入。"""

    content = str(text or "").strip()
    if not content:
        return []
    frequencies: dict[int, float] = {}
    tokens = _TOKEN_PATTERN.findall(content)
    if not tokens:
        tokens = list(content)
    for token in tokens:
        if token.strip():
            bucket = token_hash(token.casefold())
            frequencies[bucket] = frequencies.get(bucket, 0.0) + 1.0
    for character in content:
        if character.strip():
            bucket = token_hash(character)
            frequencies[bucket] = frequencies.get(bucket, 0.0) + 0.5
    norm = math.sqrt(sum(value * value for value in frequencies.values()))
    if norm <= 0:
        return []
    return sorted((bucket, value / norm) for bucket, value in frequencies.items())


def valid_embedding_item(item: object) -> bool:
    """判断一个稀疏词法特征项是否满足 schema v7 约束。"""

    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return False
    try:
        bucket = int(item[0])
        weight = float(item[1])
    except (TypeError, ValueError, OverflowError):
        return False
    return 0 <= bucket < VECTOR_DIM and math.isfinite(weight)


class SparseVectorIndex:
    """线程安全的进程内稀疏向量倒排索引。

    SQLite 仍保存完整记忆和嵌入；索引只缓存 ``bucket -> memory_id`` 的
    posting，用于把每次召回从全表余弦扫描缩小为共享特征的有限集合。
    """

    def __init__(self) -> None:
        self._vectors: dict[int, tuple[tuple[int, float], ...]] = {}
        self._postings: dict[int, dict[int, float]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _normalized_items(
        embedding: Sequence[Sequence[float | int]] | Iterable[Sequence[float | int]],
    ) -> tuple[tuple[int, float], ...]:
        merged: dict[int, float] = {}
        for item in embedding:
            if not valid_embedding_item(item):
                continue
            bucket = int(item[0])
            weight = float(item[1])
            if weight <= 0:
                continue
            merged[bucket] = merged.get(bucket, 0.0) + weight
        norm = math.sqrt(sum(weight * weight for weight in merged.values()))
        if norm <= 0 or not math.isfinite(norm):
            return ()
        return tuple(sorted((bucket, weight / norm) for bucket, weight in merged.items()))

    def replace(
        self,
        items: Iterable[
            tuple[int, Sequence[Sequence[float | int]] | Iterable[Sequence[float | int]]]
        ],
    ) -> None:
        """用持久化快照原子替换索引。"""

        vectors: dict[int, tuple[tuple[int, float], ...]] = {}
        postings: dict[int, dict[int, float]] = {}
        for raw_id, embedding in items:
            memory_id = int(raw_id)
            normalized = self._normalized_items(embedding)
            if not normalized:
                continue
            vectors[memory_id] = normalized
            for bucket, weight in normalized:
                postings.setdefault(bucket, {})[memory_id] = weight
        with self._lock:
            self._vectors = vectors
            self._postings = postings

    def upsert(
        self,
        memory_id: int,
        embedding: Sequence[Sequence[float | int]] | Iterable[Sequence[float | int]],
    ) -> None:
        """新增或替换一条记忆嵌入。"""

        normalized = self._normalized_items(embedding)
        identity = int(memory_id)
        with self._lock:
            self._remove_unlocked(identity)
            if not normalized:
                return
            self._vectors[identity] = normalized
            for bucket, weight in normalized:
                self._postings.setdefault(bucket, {})[identity] = weight

    def _remove_unlocked(self, memory_id: int) -> None:
        previous = self._vectors.pop(memory_id, ())
        for bucket, _ in previous:
            posting = self._postings.get(bucket)
            if posting is None:
                continue
            posting.pop(memory_id, None)
            if not posting:
                self._postings.pop(bucket, None)

    def remove(self, memory_id: int) -> None:
        """移除一条记忆嵌入；未知标识不会报错。"""

        with self._lock:
            self._remove_unlocked(int(memory_id))

    def remove_many(self, memory_ids: Iterable[int]) -> None:
        """在一次锁范围内移除多条记忆嵌入。"""

        with self._lock:
            for memory_id in memory_ids:
                self._remove_unlocked(int(memory_id))

    def clear(self) -> None:
        """清空可重建缓存。"""

        with self._lock:
            self._vectors.clear()
            self._postings.clear()

    def search(
        self,
        embedding: Sequence[Sequence[float | int]] | Iterable[Sequence[float | int]],
        *,
        limit: int,
    ) -> tuple[tuple[int, float], ...]:
        """返回按余弦相似度降序排列的 ``(memory_id, score)``。"""

        safe_limit = max(0, int(limit))
        query = self._normalized_items(embedding)
        if safe_limit == 0 or not query:
            return ()
        scores: dict[int, float] = {}
        with self._lock:
            for bucket, query_weight in query:
                for memory_id, memory_weight in self._postings.get(bucket, {}).items():
                    scores[memory_id] = scores.get(memory_id, 0.0) + (query_weight * memory_weight)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return tuple(ordered[:safe_limit])

    @property
    def size(self) -> int:
        """返回已索引记忆数。"""

        with self._lock:
            return len(self._vectors)

    @property
    def feature_count(self) -> int:
        """返回当前倒排特征桶数量。"""

        with self._lock:
            return len(self._postings)
