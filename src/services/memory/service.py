"""长期记忆 CRUD、检索、事件与受限 JSON 交换服务。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from core.memory_embedding import (
    VECTOR_DIM as _VECTOR_DIM,
)
from core.memory_embedding import (
    SparseVectorIndex,
    compute_embedding,
    token_hash,
    valid_embedding_item,
)
from db.database import Database
from logger.events import log_event

from .semantic import SemanticMemoryController, SemanticMemorySettings

logger = logging.getLogger(__name__)

VECTOR_DIM = _VECTOR_DIM
MAX_MEMORIES = 2000
MAX_IMPORT_BYTES = 1_000_000
MAX_IMPORT_ITEMS = 2_000
MAX_TEXT_LENGTH = 10_000
MAX_TAGS = 100
MAX_TAG_LENGTH = 128
MAX_SOURCE_LENGTH = 256
MAX_MEMORY_TYPE_LENGTH = 128

DECAY_DAYS = 7
DECAY_FLOOR_FACT = 1
DECAY_FLOOR_SUMMARY = 2
CONSOLIDATION_SIMILARITY = 0.85
PRUNE_DAYS = 30
PRUNE_IMPORTANCE_FLOOR = 1
SUMMARIZE_EVERY_N = 20
SUMMARY_CHAT_LIMIT = 20
SUMMARY_MIN_MESSAGES = 4
SUMMARY_INTERVAL_MINUTES = 360
EXCHANGE_TRUNCATE = 150
MAX_CONSOLIDATION_MEMORIES = 200
CONTEXT_MAX_CHARS = 6_000
CONTEXT_MEMORY_ITEM_CHARS = 500

# 自动提升只处理用户明确表达的稳定事实。规则保持在本地，避免把整段
# 对话发送给第二个模型，也避免模型输出被无条件写入长期记忆。
AUTO_EXTRACT_MAX_ITEMS = 3
AUTO_EXTRACT_MAX_CHARS = 240
AUTO_EXTRACT_MIN_CONFIDENCE = 0.80
AUTO_EXTRACT_DEFAULT_PRIORITY = 5
_EXTRACTION_TRIM_CHARS = " \t\r\n,，。.!！？?；;:：、~～"

_EXTRACTION_FACT_PREFIXES: dict[str, tuple[str, str]] = {
    "explicit_remember_zh": ("用户要求记住：", ""),
    "name_zh": ("用户称呼：", ""),
    "preference_dislike_zh": ("用户偏好：不喜欢", ""),
    "preference_like_zh": ("用户偏好：喜欢", ""),
    "habit_zh": ("用户习惯：", ""),
    "explicit_remember_ja": ("ユーザーが覚えておくこと：", ""),
    "preference_ja": ("ユーザーの好み：", "が好き"),
    "name_ja": ("ユーザーの名前：", ""),
    "explicit_remember_en": ("User wants remembered: ", ""),
    "name_en": ("User name: ", ""),
    "preference_en": ("User preference: likes ", ""),
}


@dataclass(frozen=True)
class ExtractedMemory:
    """规则式记忆提取结果；仅包含已通过本地安全过滤的候选。"""

    content: str
    priority: int
    rule: str
    confidence: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SummarizationBatch:
    """一次可交给模型适配器处理的有界总结批次。"""

    reason: str
    messages: tuple[Mapping[str, str], ...]
    source_ids: tuple[int, ...]
    created_at: float


_EXTRACTION_RULES: tuple[tuple[str, re.Pattern[str], int, float, tuple[str, ...]], ...] = (
    (
        "explicit_remember_zh",
        re.compile(
            r"(?:请你?|帮我)?(?:务必)?(?:记住|记下|别忘了)[：:，,\s]*(?P<value>[^。！？!?\n\r]{1,240})"
        ),
        9,
        1.0,
        ("explicit",),
    ),
    (
        "name_zh",
        re.compile(r"(?:我叫|我的名字是|称呼我为|叫我)[：:，,\s]*(?P<value>[^。！？!?\n\r]{1,80})"),
        8,
        0.95,
        ("identity",),
    ),
    (
        "preference_dislike_zh",
        re.compile(r"(?:我不喜欢|我讨厌|我不爱)[：:，,\s]*(?P<value>[^。！？!?\n\r]{1,160})"),
        7,
        0.90,
        ("preference",),
    ),
    (
        "preference_like_zh",
        re.compile(
            r"(?:我喜欢|我最喜欢|我偏好|我爱吃|我爱)[：:，,\s]*(?P<value>[^。！？!?\n\r]{1,160})"
        ),
        7,
        0.90,
        ("preference",),
    ),
    (
        "habit_zh",
        re.compile(r"(?:我习惯|我的习惯是|我通常会)[：:，,\s]*(?P<value>[^。！？!?\n\r]{1,180})"),
        6,
        0.85,
        ("habit",),
    ),
    (
        "explicit_remember_ja",
        re.compile(
            r"(?:覚えて|覚えておいて|忘れないで)[：:：，,\s]*(?P<value>[^。！？!?\n\r]{1,240})"
        ),
        9,
        1.0,
        ("explicit",),
    ),
    (
        "preference_ja",
        re.compile(r"私は(?P<value>[^。！？!?\n\r]{1,120})が好き"),
        7,
        0.90,
        ("preference",),
    ),
    (
        "name_ja",
        re.compile(r"私の名前は(?P<value>[^。！？!?\n\r]{1,80})"),
        8,
        0.95,
        ("identity",),
    ),
    (
        "explicit_remember_en",
        re.compile(
            r"\b(?:please\s+)?remember(?:\s+that)?\s+(?P<value>[^.!?\n\r]{2,240})",
            re.IGNORECASE,
        ),
        9,
        1.0,
        ("explicit",),
    ),
    (
        "name_en",
        re.compile(r"\bmy\s+name\s+is\s+(?P<value>[^.!?\n\r]{2,80})", re.IGNORECASE),
        8,
        0.95,
        ("identity",),
    ),
    (
        "preference_en",
        re.compile(r"\bI\s+(?:really\s+)?like\s+(?P<value>[^.!?\n\r]{2,160})", re.IGNORECASE),
        7,
        0.90,
        ("preference",),
    ),
)

_SENSITIVE_MEMORY_PATTERN = re.compile(
    r"(?:"
    r"(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|"
    r"auth(?:orization)?|bearer|secret|password|passwd|private[_ -]?key)"
    r"\s*(?:[:=：]|is\b|是|为)\s*\S+|"
    r"(?:密码|口令|验证码|密钥|私钥|银行卡|信用卡|身份证|社保卡|住址|门牌号|邮箱|手机号|手机号码)"
    r"|(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{16,}|xox[baprs]-[A-Za-z0-9-]{16,})"
    r"|\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    r"|(?<!\d)(?:\+?\d[\d ()-]{8,}\d)(?!\d)"
    r")",
    re.IGNORECASE,
)
_SENSITIVE_MEMORY_WORD_PATTERN = re.compile(
    r"\b(?:token|secret|password|passwd|credential|authorization|bearer|"
    r"api[_ -]?key|private[_ -]?key|email|phone|address|ssn)\b",
    re.IGNORECASE,
)

# 召回排序权重。相关性在语义索引就绪时来自真实稠密向量，否则来自明确标记
# 为词法降级的哈希稀疏特征；两者不能混称为语义嵌入。
RECALL_RELEVANCE_WEIGHT = 0.55
# 兼容旧配置/插件读取的权重名称；它表示相关性权重，不代表哈希特征是语义向量。
RECALL_SEMANTIC_WEIGHT = RECALL_RELEVANCE_WEIGHT
RECALL_EXACT_WEIGHT = 0.25
RECALL_TAG_WEIGHT = 0.05
RECALL_IMPORTANCE_WEIGHT = 0.10
RECALL_RECENCY_WEIGHT = 0.03
RECALL_ACCESS_WEIGHT = 0.02
RECALL_RECENCY_DAYS = 30.0
RECALL_ACCESS_SATURATION = 20.0
RECALL_MIN_SIMILARITY = 0.04
ALWAYS_RECALL_PRIORITY = 9


@dataclass(frozen=True)
class MemorySettings:
    """记忆运行策略。

    记忆内容仍由 SQLite 持久化；本对象只保存召回、上下文和生命周期策略，
    因而可以在配置热重载时原子替换，不需要数据库迁移。
    """

    enabled: bool = True
    recall_limit: int = 7
    context_max_chars: int = CONTEXT_MAX_CHARS
    context_memory_item_chars: int = CONTEXT_MEMORY_ITEM_CHARS
    recent_exchange_limit: int = 10
    max_memories: int = MAX_MEMORIES
    decay_days: float = DECAY_DAYS
    prune_days: float = PRUNE_DAYS
    prune_importance_floor: int = PRUNE_IMPORTANCE_FLOOR
    consolidation_enabled: bool = True
    consolidation_similarity: float = CONSOLIDATION_SIMILARITY
    max_consolidation_memories: int = MAX_CONSOLIDATION_MEMORIES
    summarize_every_n: int = SUMMARIZE_EVERY_N
    summary_chat_limit: int = SUMMARY_CHAT_LIMIT
    summarization_enabled: bool = True
    summarize_daily: bool = True
    summary_interval_minutes: int = SUMMARY_INTERVAL_MINUTES
    summary_min_messages: int = SUMMARY_MIN_MESSAGES
    exchange_min_chars: int = 20
    exchange_importance: int = 2
    exchange_decay_factor: float = 0.8
    # 自动提取是独立于 ``enabled`` 的安全开关；关闭时仍允许手动 CRUD。
    auto_extract_enabled: bool = True
    extract_max_items: int = AUTO_EXTRACT_MAX_ITEMS
    extract_max_chars: int = AUTO_EXTRACT_MAX_CHARS
    extract_min_confidence: float = AUTO_EXTRACT_MIN_CONFIDENCE
    extract_default_priority: int = AUTO_EXTRACT_DEFAULT_PRIORITY
    recall_min_similarity: float = RECALL_MIN_SIMILARITY
    always_recall_priority: int = ALWAYS_RECALL_PRIORITY
    # 召回评分权重（0~1）：相关性、精确命中、标签、持久化优先级、新鲜度、
    # 访问频次。未配置时使用编译期默认；配置值按 proportion 归一。
    recall_weights: Mapping[str, float] = field(default_factory=dict)
    # 把窗口命中率映射为相关性权重增益的有界校准：calibration_gain *
    # (1 - hit_rate)。仅服务轻微倾斜，不会覆盖其它权重维度。
    recall_calibration_gain: float = 0.06
    recall_calibration_limit: int = 4
    semantic: SemanticMemorySettings = SemanticMemorySettings()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        base_directory: str | Path | None = None,
    ) -> MemorySettings:
        """从 YAML 映射构造严格、有限的策略对象。"""

        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("memory settings must be a mapping")

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
            raise ValueError(f"memory.{name} must be a boolean")

        def integer(name: str, default: int, low: int, high: int) -> int:
            raw = value.get(name, default)
            if isinstance(raw, bool):
                raise ValueError(f"memory.{name} must be an integer")
            try:
                parsed = int(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"memory.{name} must be an integer") from exc
            if parsed != raw or parsed < low or parsed > high:
                raise ValueError(f"memory.{name} is outside the allowed range")
            return parsed

        def number(name: str, default: float, low: float, high: float) -> float:
            raw = value.get(name, default)
            if isinstance(raw, bool):
                raise ValueError(f"memory.{name} must be a number")
            try:
                parsed = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"memory.{name} must be a number") from exc
            if not math.isfinite(parsed) or parsed < low or parsed > high:
                raise ValueError(f"memory.{name} is outside the allowed range")
            return parsed

        return cls(
            enabled=boolean("enabled", True),
            recall_limit=integer("recall_limit", 7, 1, 100),
            context_max_chars=integer("context_max_chars", CONTEXT_MAX_CHARS, 512, 50_000),
            context_memory_item_chars=integer(
                "context_memory_item_chars", CONTEXT_MEMORY_ITEM_CHARS, 64, 5_000
            ),
            recent_exchange_limit=integer("recent_exchange_limit", 10, 0, 50),
            max_memories=integer("max_memories", MAX_MEMORIES, 100, 10_000),
            decay_days=number("decay_days", DECAY_DAYS, 0.1, 3_650.0),
            prune_days=number("prune_days", PRUNE_DAYS, 0.0, 3_650.0),
            prune_importance_floor=integer("prune_importance_floor", 1, 0, 10),
            consolidation_enabled=boolean("consolidation_enabled", True),
            consolidation_similarity=number("consolidation_similarity", 0.85, 0.0, 1.0),
            max_consolidation_memories=integer(
                "max_consolidation_memories", MAX_CONSOLIDATION_MEMORIES, 1, 1_000
            ),
            summarize_every_n=integer("summarize_every_n", SUMMARIZE_EVERY_N, 1, 1_000),
            summary_chat_limit=integer("summary_chat_limit", SUMMARY_CHAT_LIMIT, 4, 500),
            summarization_enabled=boolean("summarization_enabled", True),
            summarize_daily=boolean("summarize_daily", True),
            summary_interval_minutes=integer(
                "summary_interval_minutes", SUMMARY_INTERVAL_MINUTES, 0, 43_200
            ),
            summary_min_messages=integer("summary_min_messages", SUMMARY_MIN_MESSAGES, 2, 500),
            exchange_min_chars=integer("exchange_min_chars", 20, 0, 20_000),
            exchange_importance=integer("exchange_importance", 2, 0, 10),
            exchange_decay_factor=number("exchange_decay_factor", 0.8, 0.0, 10.0),
            auto_extract_enabled=boolean("auto_extract_enabled", True),
            extract_max_items=integer("extract_max_items", AUTO_EXTRACT_MAX_ITEMS, 0, 20),
            extract_max_chars=integer("extract_max_chars", AUTO_EXTRACT_MAX_CHARS, 32, 2_000),
            extract_min_confidence=number(
                "extract_min_confidence", AUTO_EXTRACT_MIN_CONFIDENCE, 0.0, 1.0
            ),
            extract_default_priority=integer(
                "extract_default_priority", AUTO_EXTRACT_DEFAULT_PRIORITY, 0, 10
            ),
            recall_min_similarity=number("recall_min_similarity", RECALL_MIN_SIMILARITY, 0.0, 1.0),
            always_recall_priority=integer("always_recall_priority", ALWAYS_RECALL_PRIORITY, 0, 10),
            recall_weights=_recall_weights(value.get("recall_weights")),
            recall_calibration_gain=number("recall_calibration_gain", 0.06, 0.0, 0.25),
            recall_calibration_limit=integer("recall_calibration_limit", 4, 1, 256),
            semantic=SemanticMemorySettings.from_mapping(
                value.get("semantic"),
                base_directory=base_directory,
            ),
        )


def _recall_weights(value: object) -> dict[str, float]:
    """把配置权重转为严格键名下的有限比例集合。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("memory.recall_weights must be a mapping")
    allowed = {
        "relevance",
        "exact",
        "tag",
        "importance",
        "recency",
        "access",
    }
    unsupported = tuple(key for key in value if str(key).strip() not in allowed)
    if unsupported:
        raise ValueError("memory.recall_weights contains unsupported fields")
    result: dict[str, float] = {}
    for key in allowed:
        raw = value.get(key)
        if raw is None:
            continue
        if isinstance(raw, bool):
            raise ValueError(f"memory.recall_weights.{key} must be a number")
        try:
            parsed = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"memory.recall_weights.{key} must be a number") from exc
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"memory.recall_weights.{key} is outside the allowed range")
        result[key] = parsed
    return result


def _compute_embedding(text: str | None) -> list[tuple[int, float]]:
    """保持记忆服务的旧内部入口，同时复用核心嵌入实现。"""

    return compute_embedding(text)


def _token_hash(token: str) -> int:
    """保持旧内部入口。"""

    return token_hash(token)


def _valid_embedding_item(item: object) -> bool:
    """保持旧内部入口。"""

    return valid_embedding_item(item)


_CONTRADICTION_PAIRS = (
    ("喜欢", "讨厌"),
    ("想要", "不想要"),
    ("想", "不想"),
    ("能", "不能"),
    ("可以", "不可以"),
    ("愿意", "不愿意"),
)


def _cosine_similarity_sparse(
    left: Sequence[tuple[int, float]],
    right: Sequence[tuple[int, float]],
) -> float:
    if not left or not right:
        return 0.0
    right_values = dict(right)
    return sum(value * right_values.get(bucket, 0.0) for bucket, value in left)


def _content_hash(content: object) -> str:
    return hashlib.sha256(str(content or "").strip().encode("utf-8")).hexdigest()


def _json_load(value: object, default: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default
    return value


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(cast(float | int | str | bytes, value))
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(cast(int | float | str | bytes, value))
    except (TypeError, ValueError, OverflowError):
        return default


def _finite_nonnegative(value: object, field_name: str) -> float:
    """校验需要写入数据库的非负浮点值，拒绝 NaN/Inf。"""

    try:
        parsed = float(cast(float | int | str | bytes, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return parsed


def _normalize_priority(value: object, field_name: str = "importance") -> int:
    """将兼容字段 ``importance``/``priority`` 归一化到 0-10。"""

    try:
        parsed = int(cast(int | float | str | bytes, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    return max(0, min(10, parsed))


def _memory_timeframe(item: Mapping[str, object]) -> str:
    """把记忆条目时间前景转成提示词中的时间段标签。

    优先读取 ``created``/``updated`` 的 Unix 秒级时间戳（3 天窗口），
    其次读取 ``last_recalled``；解析失败或超窗返回空串，调用方跳过。
    只输出稳定短标签，不把时间戳或来源字段带入提示词。
    """

    for key in ("created", "updated"):
        raw = item.get(key)
        try:
            value = float(cast(float | int | str | bytes, raw))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and value > 0 and time.time() - value <= 3 * 86400:
            return "近期"
    recalled = item.get("last_recalled")
    if recalled is not None:
        try:
            if math.isfinite(float(recalled)) and time.time() - float(recalled) <= 3 * 86400:
                return "近期想起"
        except (TypeError, ValueError, OverflowError):
            pass
    return ""


@dataclass(frozen=True)
class Memory:
    """不向调用方暴露 SQLite 行对象的记忆值对象。

    ``importance`` 是数据库兼容字段；``priority`` 作为只读别名供新编排器
    表达“召回优先级”，两者始终保持一致。
    """

    id: int
    content: str
    importance: int
    source: str
    tags: tuple[str, ...]
    metadata: dict[str, Any]
    memory_type: str
    created: float
    updated: float
    decay_factor: float = 1.0
    source_ids: tuple[int, ...] = ()
    last_recalled: float = 0.0
    access_count: int = 1
    embedding: tuple[tuple[int, float], ...] = ()
    last_decay: float = 0.0

    @property
    def priority(self) -> int:
        """返回与旧 ``importance`` 字段一致的持久化优先级。"""

        return self.importance


class MemoryService:
    """隐藏 SQL 与 JSON 细节的长期记忆服务。"""

    def __init__(
        self,
        database: Database,
        affection: object | None = None,
        settings: MemorySettings | Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(database, Database):
            raise TypeError("database must be a Database")
        self._database = database
        self._affection = affection
        self._settings = (
            settings
            if isinstance(settings, MemorySettings)
            else MemorySettings.from_mapping(settings)
        )
        self._vector_index = SparseVectorIndex()
        self._rebuild_vector_index()
        self._fts_available = self._detect_fts5()
        # 构造器只启动轻量后台协调线程；模型导入、加载和 ANN 重建均不在
        # Qt 首窗/asyncio 线程同步执行。
        self._semantic = SemanticMemoryController(
            database,
            self._settings.semantic,
            capacity=self._settings.max_memories,
        )
        # 同一模型进程只允许一个上下文查询；重建/锁占用时其它回合直接走
        # 词法路径，避免创建无界等待线程。
        self._semantic_context_slot = threading.BoundedSemaphore(1)
        self._semantic_context_lifecycle_lock = threading.RLock()
        self._semantic_context_threads: set[threading.Thread] = set()
        self._semantic_context_cancellations: set[threading.Event] = set()
        self._semantic_context_closing = False
        # 调用窗口召回统计：记录最近 32 次的(命中, 首项分数)。
        self._recall_stats_lock = threading.Lock()
        self._recall_stats_window: list[tuple[bool, float]] = []
        self._recall_stats_max_entries = 32

    @property
    def settings(self) -> MemorySettings:
        """返回当前不可变策略快照。"""

        return self._settings

    @property
    def enabled(self) -> bool:
        """返回是否将记忆接入对话上下文和自动写入。"""

        return bool(self._settings.enabled)

    def configure(self, settings: MemorySettings | Mapping[str, Any] | None) -> MemorySettings:
        """原子替换记忆策略，返回应用后的快照。"""

        next_settings = (
            settings
            if isinstance(settings, MemorySettings)
            else MemorySettings.from_mapping(settings)
        )
        self._settings = next_settings
        with self._database.transaction() as connection:
            removed_ids = self._enforce_cap(connection)
        self._vector_index.remove_many(removed_ids)
        self._semantic.configure(
            next_settings.semantic,
            capacity=next_settings.max_memories,
        )
        if removed_ids:
            self._semantic.notify_mutation()
        return next_settings

    def status(self) -> dict[str, object]:
        """返回不含内容的记忆策略状态，供控制台/诊断使用。"""

        value = self._settings
        semantic = self._semantic.status()
        recall = self.recall_statistics()
        return {
            "enabled": value.enabled,
            "recall_limit": value.recall_limit,
            "context_max_chars": value.context_max_chars,
            "recent_exchange_limit": value.recent_exchange_limit,
            "max_memories": value.max_memories,
            "decay_days": value.decay_days,
            "prune_days": value.prune_days,
            "consolidation_enabled": value.consolidation_enabled,
            "summarization_enabled": value.summarization_enabled,
            "summarize_daily": value.summarize_daily,
            "summary_interval_minutes": value.summary_interval_minutes,
            "summary_min_messages": value.summary_min_messages,
            "auto_extract_enabled": value.auto_extract_enabled,
            "extract_max_items": value.extract_max_items,
            "extract_max_chars": value.extract_max_chars,
            "extract_min_confidence": value.extract_min_confidence,
            "extract_default_priority": value.extract_default_priority,
            "recall_min_similarity": value.recall_min_similarity,
            "always_recall_priority": value.always_recall_priority,
            "vector_index_size": self._vector_index.size,
            "vector_index_features": self._vector_index.feature_count,
            "lexical_index": "fts5" if self._fts_available else "sparse_fallback",
            "sparse_index_kind": "deterministic_hash_lexical",
            "semantic_enabled": semantic["enabled"],
            "semantic_index": semantic["status"],
            "semantic_degraded_reason": semantic["degraded_reason"],
            "semantic_provider": semantic["provider"],
            "semantic_model_id": semantic["model_id"],
            "semantic_model_revision": semantic["model_revision"],
            "semantic_dimension": semantic["dimension"],
            "semantic_batch_size": semantic["batch_size"],
            "semantic_timeout_seconds": semantic["timeout_seconds"],
            "semantic_index_size": semantic["index_size"],
            "semantic_pending_mutations": semantic["pending_mutations"],
            "semantic_active_generation": semantic["active_generation"],
            "semantic_active_model_id": semantic["active_model_id"],
            "semantic_active_model_revision": semantic["active_model_revision"],
            "semantic_active_package_version": semantic["active_package_version"],
            "recall_total": recall["total"],
            "recall_hit_rate": recall["hit_rate"],
            "recall_top_mean_score": recall["top_mean_score"],
            "recall_calibration": self.recall_calibration(),
        }

    def recall_statistics(self) -> dict[str, float | int]:
        """返回最近调用窗口的召回命中率与首项平均分。

        统计在每次召回时经简单有环窗口更新，命中定义为“返回至少一条
        similarity/exact/tag 任一非零的记忆”；窗口大小固定为 32 次，
        避免跨日累积拖慢旧统计快照。
        """

        with self._recall_stats_lock:
            outcomes = list(self._recall_stats_window)
        total = len(outcomes)
        if not total:
            return {"total": 0, "hit_rate": 0.0, "top_mean_score": 0.0}
        hits = sum(1 for outcome in outcomes if outcome[0])
        scores = [outcome[1] for outcome in outcomes if math.isfinite(outcome[1])]
        mean = sum(scores) / len(scores) if scores else 0.0
        return {
            "total": int(total),
            "hit_rate": round(hits / total, 4),
            "top_mean_score": round(max(0.0, min(1.0, mean)), 4),
        }

    def _rebuild_vector_index(self) -> None:
        """从 SQLite 快照重建可丢弃的进程内索引。"""

        with self._database._lock:
            rows = self._database.connection.execute(
                "SELECT id, embedding FROM memories ORDER BY id"
            ).fetchall()
        self._vector_index.replace(
            (int(row["id"]), self._parse_emb(row["embedding"])) for row in rows
        )

    def _detect_fts5(self) -> bool:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
            ).fetchone()
        return row is not None

    @staticmethod
    def _memory_is_active(memory: Memory) -> bool:
        """被更新事实取代的历史记录保留审计能力，但不再进入对话召回。"""

        return not bool(memory.metadata.get("superseded_by"))

    @staticmethod
    def _safe_tags(tags: object) -> tuple[str, ...]:
        if tags is None:
            return ()
        if isinstance(tags, str) or not isinstance(tags, Sequence):
            raise ValueError("memory tags must be a sequence")
        result: list[str] = []
        for tag in tags:
            value = str(tag).strip() if isinstance(tag, str) else ""
            if not value:
                continue
            if len(value) > MAX_TAG_LENGTH:
                raise ValueError("memory tag is too long")
            if value not in result:
                result.append(value)
        if len(result) > MAX_TAGS:
            raise ValueError("too many memory tags")
        return tuple(result)

    @staticmethod
    def _safe_source(value: object) -> str:
        """规范化来源字段并拒绝超长值与控制字符。"""

        result = str(value or "").strip()
        if len(result) > MAX_SOURCE_LENGTH:
            raise ValueError("memory source is too long")
        if any(character in result for character in "\x00\r\n"):
            raise ValueError("memory source is invalid")
        return result

    @staticmethod
    def _safe_memory_type(value: object) -> str:
        """规范化记忆类型并拒绝超长值与控制字符。"""

        result = str(value or "fact").strip() or "fact"
        if len(result) > MAX_MEMORY_TYPE_LENGTH:
            raise ValueError("memory_type is too long")
        if any(character in result for character in "\x00\r\n"):
            raise ValueError("memory_type is invalid")
        return result

    @staticmethod
    def _safe_metadata(metadata: object) -> dict[str, Any]:
        if metadata is None:
            return {}
        if not isinstance(metadata, Mapping):
            raise ValueError("memory metadata must be a mapping")
        try:
            encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("memory metadata is not JSON serializable") from exc
        if len(encoded.encode("utf-8")) > MAX_TEXT_LENGTH:
            raise ValueError("memory metadata is too large")
        return dict(metadata)

    @staticmethod
    def _safe_source_ids(source_ids: object) -> tuple[int, ...]:
        if source_ids is None:
            return ()
        if isinstance(source_ids, (str, bytes, bytearray)) or not isinstance(source_ids, Sequence):
            raise ValueError("source_ids must be a sequence")
        values: list[int] = []
        for item in source_ids:
            # 来源编号用于关联 SQLite 行；静默把 1.5、True 或任意对象
            # 转成整数会破坏溯源关系，因此只接受明确的整数表示。
            if isinstance(item, bool):
                raise ValueError("source_ids must contain integers")
            if isinstance(item, int):
                value = item
            elif isinstance(item, float):
                if not math.isfinite(item) or not item.is_integer():
                    raise ValueError("source_ids must contain integers")
                value = int(item)
            elif isinstance(item, str):
                text = item.strip()
                if not re.fullmatch(r"[+-]?\d+", text):
                    raise ValueError("source_ids must contain integers")
                value = int(text)
            else:
                raise ValueError("source_ids must contain integers")
            if value < -(2**63) or value > (2**63 - 1):
                raise ValueError("source_id is outside the SQLite integer range")
            values.append(value)
        if len(values) > 500:
            raise ValueError("too many source_ids")
        return tuple(values)

    @classmethod
    def _from_row(cls, row: Any) -> Memory:
        tags = _json_load(row["tags"], [])
        metadata = _json_load(row["metadata"], {})
        source_ids = _json_load(row["source_ids"], [])
        embedding = _json_load(row["embedding"], [])
        safe_tags = (
            tuple(item for item in tags if isinstance(item, str)) if isinstance(tags, list) else ()
        )
        safe_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        if isinstance(source_ids, list):
            try:
                safe_source_ids = tuple(int(item) for item in source_ids)
            except (TypeError, ValueError, OverflowError):
                safe_source_ids = ()
        else:
            safe_source_ids = ()
        safe_embedding = (
            tuple(
                (int(item[0]), float(item[1]))
                for item in embedding
                if isinstance(item, (list, tuple))
                and len(item) == 2
                and _valid_embedding_item(item)
            )
            if isinstance(embedding, list)
            else ()
        )
        return Memory(
            id=_safe_int(row["id"]),
            content=str(row["content"] or ""),
            importance=_safe_int(
                row["importance"] if row["importance"] is not None else 1,
                1,
            ),
            source=str(row["source"] or ""),
            tags=safe_tags,
            metadata=safe_metadata,
            memory_type=str(row["memory_type"] or "fact"),
            created=_safe_float(row["created"] or 0.0),
            updated=_safe_float(row["updated"] or row["created"] or 0.0),
            decay_factor=_safe_float(
                row["decay_factor"] if row["decay_factor"] is not None else 1.0,
                1.0,
            ),
            source_ids=safe_source_ids,
            last_recalled=_safe_float(row["last_recalled"] or 0.0),
            access_count=_safe_int(
                row["access_count"] if row["access_count"] is not None else 1,
                1,
            ),
            embedding=safe_embedding,
            last_decay=_safe_float(row["last_decay"] or 0.0),
        )

    @classmethod
    def _memory_to_dict(cls, memory: Memory) -> dict[str, Any]:
        """把已读取的值对象转换为兼容字典，避免重复访问 SQLite。"""

        return {
            "id": memory.id,
            "content": memory.content,
            "importance": memory.importance,
            "priority": memory.priority,
            "source": memory.source,
            "created": memory.created,
            "last_recalled": memory.last_recalled,
            "tags": list(memory.tags),
            "metadata": dict(memory.metadata),
            "memory_type": memory.memory_type,
            "embedding": [list(item) for item in memory.embedding],
            "decay_factor": memory.decay_factor,
            "updated": memory.updated,
            "access_count": memory.access_count,
            "source_ids": list(memory.source_ids),
            "last_decay": memory.last_decay,
        }

    @classmethod
    def _row_to_dict(cls, row: Any) -> dict[str, Any]:
        return cls._memory_to_dict(cls._from_row(row))

    def _insert_memory_unlocked(
        self,
        connection: Any,
        *,
        content: str,
        importance: int,
        source: str,
        tags: Sequence[str],
        metadata: Mapping[str, Any],
        memory_type: str,
        source_ids: Sequence[int],
        decay_factor: float,
        now: float,
    ) -> tuple[Memory, tuple[int, ...]]:
        """在调用方事务内插入已校验记忆，并返回容量淘汰结果。"""

        embedding = _compute_embedding(content)
        cursor = connection.execute(
            """
            INSERT INTO memories (
                content, importance, source, created, last_recalled, tags, metadata,
                memory_type, embedding, decay_factor, updated, access_count, source_ids,
                last_decay
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content,
                importance,
                source,
                now,
                now,
                json.dumps(tuple(tags), ensure_ascii=False),
                json.dumps(dict(metadata), ensure_ascii=False),
                memory_type,
                json.dumps(embedding, ensure_ascii=False),
                decay_factor,
                now,
                1,
                json.dumps(tuple(source_ids), ensure_ascii=False),
                now,
            ),
        )
        row = connection.execute(
            "SELECT * FROM memories WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        created = self._from_row(row)
        return created, self._enforce_cap(connection, protected_ids=(created.id,))

    def create(
        self,
        content: str,
        *,
        importance: int = 1,
        source: str = "",
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        memory_type: str = "fact",
        source_ids: Sequence[int] = (),
        decay_factor: float = 1.0,
        priority: int | None = None,
    ) -> Memory:
        text = str(content or "").strip()
        if not text or len(text) > MAX_TEXT_LENGTH:
            raise ValueError("memory content must contain 1 to 10000 characters")
        safe_tags = self._safe_tags(tags)
        safe_metadata = self._safe_metadata(metadata)
        safe_source_ids = self._safe_source_ids(source_ids)
        factor = _finite_nonnegative(decay_factor, "decay_factor")
        effective_importance = importance if priority is None else priority
        normalized_importance = _normalize_priority(
            effective_importance, "priority" if priority is not None else "importance"
        )
        now = time.time()
        with self._database.transaction() as connection:
            created, removed_ids = self._insert_memory_unlocked(
                connection,
                content=text,
                importance=normalized_importance,
                source=self._safe_source(source),
                tags=safe_tags,
                metadata=safe_metadata,
                memory_type=self._safe_memory_type(memory_type),
                source_ids=safe_source_ids,
                decay_factor=factor,
                now=now,
            )
        self._vector_index.upsert(created.id, created.embedding)
        self._vector_index.remove_many(removed_ids)
        self._semantic.notify_mutation()
        log_event(
            logger,
            "memory.record.created",
            component="memory",
            status="completed",
            fields={
                "importance": normalized_importance,
                "tag_count": len(safe_tags),
                "source_count": len(safe_source_ids),
                "evicted_count": len(removed_ids),
            },
        )
        return created

    @staticmethod
    def _normalize_extracted_content(value: object, limit: int) -> str:
        """压缩空白并去除边界标点，返回有界的记忆文本。"""

        text = " ".join(str(value or "").split()).strip(_EXTRACTION_TRIM_CHARS)
        return text[: max(1, int(limit))].rstrip(_EXTRACTION_TRIM_CHARS).strip()

    @staticmethod
    def _contains_sensitive_memory_data(value: object) -> bool:
        """拒绝凭据、联系方式和高风险身份信息，且不尝试脱敏后持久化。"""

        text = str(value or "").strip()
        if not text:
            return False
        return bool(
            _SENSITIVE_MEMORY_PATTERN.search(text) or _SENSITIVE_MEMORY_WORD_PATTERN.search(text)
        )

    def extract_candidates(self, text: str) -> tuple[ExtractedMemory, ...]:
        """从用户明确表达中提取有限候选，不访问模型、不写入数据库。

        只匹配稳定事实/明确记忆请求；候选先经过长度、置信度和敏感信息
        过滤，再按优先级排序并去重。这样该步骤可在回合完成后快速同步执行。
        """

        settings = self._settings
        if not self.enabled or not settings.auto_extract_enabled or settings.extract_max_items <= 0:
            return ()
        source_text = str(text or "").strip()
        if not source_text:
            return ()
        # 对超长输入只检查有限前缀，避免规则式扫描占用主线程过久；对话入口
        # 本身也有 20,000 字上限，因此这里保留同一边界。
        source_text = source_text[:20_000]
        candidates: list[tuple[int, ExtractedMemory]] = []
        seen: set[str] = set()
        for rule, pattern, rule_priority, confidence, rule_tags in _EXTRACTION_RULES:
            if confidence < settings.extract_min_confidence:
                continue
            for match in pattern.finditer(source_text):
                raw_value = match.groupdict().get("value") or match.group(0)
                # 命名组只保留谓词后的事实；中文逗号通常表示下一条子句，
                # 在这里截断可避免把多个事实拼成一条身份记忆。
                raw_value = re.split(r"[,，、;；]", raw_value, maxsplit=1)[0]
                value = self._normalize_extracted_content(raw_value, settings.extract_max_chars)
                prefix, suffix = _EXTRACTION_FACT_PREFIXES.get(rule, ("用户事实：", ""))
                content = self._normalize_extracted_content(
                    f"{prefix}{value}{suffix}", settings.extract_max_chars
                )
                if len(content) < 4 or self._contains_sensitive_memory_data(content):
                    continue
                key = " ".join(content.casefold().split())
                if not key or key in seen:
                    continue
                seen.add(key)
                priority = _normalize_priority(
                    max(rule_priority, settings.extract_default_priority),
                    "priority",
                )
                tags = tuple(dict.fromkeys(("auto-extracted", *rule_tags)))
                candidates.append(
                    (
                        match.start(),
                        ExtractedMemory(content, priority, rule, confidence, tags),
                    )
                )
        candidates.sort(key=lambda item: (-item[1].priority, -item[1].confidence, item[0]))
        return tuple(item[1] for item in candidates[: settings.extract_max_items])

    @staticmethod
    def _extraction_key(content: object) -> str:
        """生成仅用于比较的稳定文本键，不改变持久化原文。"""

        return " ".join(str(content or "").casefold().split()).strip(_EXTRACTION_TRIM_CHARS)

    def _find_extraction_duplicate(
        self,
        content: str,
        *,
        memory_type: str = "fact",
    ) -> Memory | None:
        """优先找规范化完全相同的记忆；不触发召回计数。"""

        key = self._extraction_key(content)
        if not key:
            return None
        with self._database._lock:
            rows = self._database.connection.execute(
                "SELECT * FROM memories WHERE memory_type = ? "
                "ORDER BY importance DESC, updated DESC, id DESC",
                (str(memory_type or "fact"),),
            ).fetchall()
        for row in rows:
            if self._extraction_key(row["content"]) == key:
                return self._from_row(row)
        return None

    def _promote_candidate(
        self,
        candidate: ExtractedMemory,
        *,
        source: str = "conversation:auto_extract",
        source_ids: Sequence[int] = (),
    ) -> Memory | None:
        """创建或提升单条候选；任何安全校验失败都返回 None。"""

        content = self._normalize_extracted_content(
            candidate.content, self._settings.extract_max_chars
        )
        if len(content) < 4 or self._contains_sensitive_memory_data(content):
            return None
        try:
            safe_source = self._safe_source(source)
            safe_source_ids = self._safe_source_ids(source_ids)
        except (TypeError, ValueError, OverflowError):
            return None
        now = time.time()
        duplicate = self._find_extraction_duplicate(content)
        metadata = {
            "auto_extracted": True,
            "extraction_rule": str(candidate.rule or "rule"),
            "extraction_confidence": float(candidate.confidence),
            "last_extracted_at": now,
        }
        if duplicate is not None:
            merged_metadata = dict(duplicate.metadata)
            try:
                extraction_count = int(merged_metadata.get("extraction_count", 0))
            except (TypeError, ValueError, OverflowError):
                extraction_count = 0
            merged_metadata.update(metadata)
            next_count = min(1000, max(0, extraction_count) + 1)
            merged_metadata["extraction_count"] = next_count
            merged_tags = tuple(dict.fromkeys((*duplicate.tags, *candidate.tags)))
            merged_source_ids = tuple(dict.fromkeys((*duplicate.source_ids, *safe_source_ids)))[
                :500
            ]
            reinforcement = min(3, int(math.log2(max(1, next_count))))
            promoted_priority = max(
                duplicate.importance,
                _normalize_priority(candidate.priority) + reinforcement,
            )
            return self.update(
                duplicate.id,
                priority=promoted_priority,
                tags=merged_tags,
                metadata=merged_metadata,
                source_ids=merged_source_ids,
                source=duplicate.source or safe_source,
            )
        try:
            created = self.create(
                content,
                priority=_normalize_priority(candidate.priority),
                source=safe_source,
                tags=candidate.tags,
                metadata={**metadata, "extraction_count": 1},
                memory_type="fact",
                source_ids=safe_source_ids,
            )
            self._supersede_identity_memories(created)
            return created
        except (TypeError, ValueError, OverflowError):
            return None

    def _supersede_identity_memories(self, current: Memory) -> int:
        """让新的姓名事实立即取代旧姓名，同时保留旧记录用于审计。"""

        if "identity" not in current.tags:
            return 0
        changed = 0
        now = time.time()
        for memory in self.list(limit=self._settings.max_memories, memory_type="fact"):
            if (
                memory.id == current.id
                or "identity" not in memory.tags
                or not self._memory_is_active(memory)
            ):
                continue
            metadata = dict(memory.metadata)
            metadata.update({"superseded_by": current.id, "superseded_at": now})
            if (
                self.update(
                    memory.id,
                    metadata=metadata,
                    priority=max(0, memory.importance - 2),
                )
                is not None
            ):
                changed += 1
        return changed

    def extract_and_promote(
        self,
        text: str,
        *,
        source: str = "conversation:auto_extract",
        source_ids: Sequence[int] = (),
    ) -> tuple[Memory, ...]:
        """提取并创建/提升用户明确事实，返回实际受影响的记忆。

        该接口默认遵循 ``memory.enabled`` 与 ``memory.auto_extract_enabled``；
        手动 CRUD 不受此开关影响。候选数量和文本长度均受 ``MemorySettings``
        限制，重复候选只提升既有优先级而不会新增行。
        """

        if not self.enabled or not self._settings.auto_extract_enabled:
            return ()
        result: list[Memory] = []
        for candidate in self.extract_candidates(text):
            memory = self._promote_candidate(candidate, source=source, source_ids=source_ids)
            if memory is not None and all(item.id != memory.id for item in result):
                result.append(memory)
        return tuple(result)

    def promote_extracted(
        self,
        content: str,
        *,
        priority: int | None = None,
        source: str = "conversation:auto_extract",
        tags: Sequence[str] = ("auto-extracted",),
        rule: str = "manual",
        confidence: float = 1.0,
        source_ids: Sequence[int] = (),
    ) -> Memory | None:
        """显式提升一条已由调用方确认的记忆，仍执行敏感信息拒绝和去重。"""

        if not self.enabled:
            return None
        try:
            safe_confidence = float(confidence)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            not math.isfinite(safe_confidence)
            or safe_confidence < self._settings.extract_min_confidence
        ):
            return None
        candidate = ExtractedMemory(
            self._normalize_extracted_content(content, self._settings.extract_max_chars),
            _normalize_priority(
                self._settings.extract_default_priority if priority is None else priority,
                "priority",
            ),
            str(rule or "manual")[:128],
            safe_confidence,
            tuple(tags),
        )
        return self._promote_candidate(candidate, source=source, source_ids=source_ids)

    def promote_extracted_batch(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        source: str = "conversation:model_extract",
        source_ids: Sequence[int] = (),
    ) -> tuple[Memory, ...]:
        """接收模型适配器的结构化结果，并复用本地安全过滤与即时索引。"""

        if not self.enabled or isinstance(items, (str, bytes, bytearray)):
            return ()
        promoted: list[Memory] = []
        for item in tuple(items)[: self._settings.extract_max_items]:
            if not isinstance(item, Mapping):
                continue
            tags = item.get("tags", ("auto-extracted", "model-extracted"))
            if isinstance(tags, str) or not isinstance(tags, Sequence):
                continue
            try:
                memory = self.promote_extracted(
                    str(item.get("content") or ""),
                    priority=(None if item.get("priority") is None else int(item["priority"])),
                    source=source,
                    tags=tuple(str(tag) for tag in tags),
                    rule=str(item.get("rule") or "model")[:128],
                    confidence=float(item.get("confidence", 1.0)),
                    source_ids=source_ids,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            if memory is not None and all(existing.id != memory.id for existing in promoted):
                promoted.append(memory)
        return tuple(promoted)

    def get(self, memory_id: int) -> Memory | None:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT * FROM memories WHERE id = ?", (int(memory_id),)
            ).fetchone()
        return self._from_row(row) if row else None

    def update(
        self,
        memory_id: int,
        *,
        content: str | None = None,
        importance: int | None = None,
        source: str | None = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        memory_type: str | None = None,
        source_ids: Sequence[int] | None = None,
        decay_factor: float | None = None,
        priority: int | None = None,
    ) -> Memory | None:
        current = self.get(memory_id)
        if current is None:
            return None
        text = current.content if content is None else str(content or "").strip()
        if not text or len(text) > MAX_TEXT_LENGTH:
            raise ValueError("memory content must contain 1 to 10000 characters")
        safe_tags = current.tags if tags is None else self._safe_tags(tags)
        safe_metadata = current.metadata if metadata is None else self._safe_metadata(metadata)
        safe_source_ids = (
            current.source_ids if source_ids is None else self._safe_source_ids(source_ids)
        )
        factor = (
            current.decay_factor
            if decay_factor is None
            else _finite_nonnegative(decay_factor, "decay_factor")
        )
        # ``priority`` 是新 API 的显式别名；两个字段同时提供时优先采用它，
        # 否则必须保留既有 ``importance`` 更新语义，不能悄悄回退为旧值。
        effective_importance = (
            priority
            if priority is not None
            else current.importance
            if importance is None
            else importance
        )
        normalized_importance = _normalize_priority(
            effective_importance, "priority" if priority is not None else "importance"
        )
        now = time.time()
        embedding = _compute_embedding(text)
        with self._database.transaction() as connection:
            connection.execute(
                """
                UPDATE memories SET content = ?, importance = ?, source = ?, tags = ?,
                    metadata = ?, memory_type = ?, embedding = ?, decay_factor = ?,
                    source_ids = ?, updated = ? WHERE id = ?
                """,
                (
                    text,
                    normalized_importance
                    if (importance is not None or priority is not None)
                    else current.importance,
                    current.source if source is None else self._safe_source(source),
                    json.dumps(safe_tags, ensure_ascii=False),
                    json.dumps(safe_metadata, ensure_ascii=False),
                    (
                        current.memory_type
                        if memory_type is None
                        else self._safe_memory_type(memory_type)
                    ),
                    json.dumps(embedding, ensure_ascii=False),
                    factor,
                    json.dumps(safe_source_ids, ensure_ascii=False),
                    now,
                    int(memory_id),
                ),
            )
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?", (int(memory_id),)
            ).fetchone()
        updated = self._from_row(row) if row else None
        if updated is not None:
            self._vector_index.upsert(updated.id, updated.embedding)
            self._semantic.notify_mutation()
        return updated

    def delete(self, memory_id: int) -> bool:
        with self._database.transaction() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (int(memory_id),))
        deleted = cursor.rowcount > 0
        if deleted:
            self._vector_index.remove(int(memory_id))
            self._semantic.notify_mutation()
        return deleted

    def list(
        self,
        *,
        limit: int = 100,
        tag: str | None = None,
        memory_type: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> tuple[Memory, ...]:
        safe_limit = max(1, min(int(limit), self._settings.max_memories))
        conditions: list[str] = []
        params: list[object] = []
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(self._safe_memory_type(memory_type))
        filter_tags = list(tags or ())
        if tag:
            filter_tags.append(tag)
        if filter_tags:
            conditions.append(
                "("
                + " OR ".join(
                    "EXISTS (SELECT 1 FROM json_each(CASE WHEN json_valid(memories.tags) "
                    "THEN memories.tags ELSE '[]' END) "
                    "WHERE json_each.type = 'text' AND json_each.value = ?)"
                    for _ in filter_tags
                )
                + ")"
            )
            params.extend(str(item) for item in filter_tags)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._database._lock:
            rows = self._database.connection.execute(
                f"SELECT * FROM memories{where} ORDER BY importance DESC, updated DESC LIMIT ?",
                (*params, safe_limit),
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    @staticmethod
    def _recall_score(
        memory: Memory,
        *,
        similarity: float,
        exact: float,
        tag_match: float,
        now: float,
        weights: Mapping[str, float] | None = None,
        calibration: float = 0.0,
    ) -> float:

        configured = dict(weights or {})
        total_configured = sum(configured.values())
        proportion = {
            "relevance": configured.get("relevance", RECALL_RELEVANCE_WEIGHT),
            "exact": configured.get("exact", RECALL_EXACT_WEIGHT),
            "tag": configured.get("tag", RECALL_TAG_WEIGHT),
            "importance": configured.get("importance", RECALL_IMPORTANCE_WEIGHT),
            "recency": configured.get("recency", RECALL_RECENCY_WEIGHT),
            "access": configured.get("access", RECALL_ACCESS_WEIGHT),
        }
        if total_configured > 0 and math.isfinite(total_configured):
            scale = 1.0 / total_configured
            proportion = {key: value * scale for key, value in proportion.items()}
        boost = max(0.0, min(0.4, float(calibration or 0.0)))
        if boost > 0:
            moved = proportion["relevance"] * boost
            keep_ratio = 1.0 - min(
                1.0, moved / max(proportion["importance"] + proportion["access"], 1e-9)
            )
            proportion["relevance"] += moved
            proportion["importance"] *= keep_ratio
            proportion["access"] *= keep_ratio
        priority = max(0.0, min(10.0, _safe_float(memory.importance, 0.0))) / 10.0
        # ``updated`` 也会在衰减维护时写入，不能把维护动作误当成近期使用。
        anchor = max(_safe_float(memory.created), _safe_float(memory.last_recalled))
        age_days = max(0.0, (now - anchor) / 86400.0)
        recency = 1.0 / (1.0 + age_days / RECALL_RECENCY_DAYS)
        decay_factor = _safe_float(memory.decay_factor, 1.0)
        retention = 1.0 if decay_factor <= 0 else min(1.0, decay_factor)
        recency *= retention
        access_count = max(0.0, _safe_float(memory.access_count, 0.0))
        access = min(1.0, math.log1p(access_count) / math.log1p(RECALL_ACCESS_SATURATION))
        score = (
            max(0.0, min(1.0, similarity)) * proportion["relevance"]
            + max(0.0, min(1.0, exact)) * proportion["exact"]
            + max(0.0, min(1.0, tag_match)) * proportion["tag"]
            + priority * proportion["importance"]
            + recency * proportion["recency"]
            + access * proportion["access"]
        )
        return score if math.isfinite(score) else 0.0

    def recall_calibration(self) -> float:
        """把窗口命中率映射为 0~1 的相关性校准增益。"""

        stats = self.recall_statistics()
        total = int(stats.get("total", 0) or 0)
        if total < self._settings.recall_calibration_limit:
            return 0.0
        hit_rate = float(stats.get("hit_rate", 0.0) or 0.0)
        if not math.isfinite(hit_rate):
            return 0.0
        return min(
            self._settings.recall_calibration_gain,
            max(0.0, self._settings.recall_calibration_gain * (1.0 - hit_rate)),
        )

    def _fts_candidate_ids(self, query: str, limit: int) -> tuple[int, ...]:
        """读取 FTS5 候选；扩展缺失或运行时失效时返回空集合。"""

        text = str(query or "").strip()
        if not self._fts_available or not text or limit <= 0:
            return ()
        expression = f'"{text.replace(chr(34), chr(34) * 2)}"'
        try:
            with self._database._lock:
                rows = self._database.connection.execute(
                    "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ? "
                    "ORDER BY bm25(memory_fts) LIMIT ?",
                    (expression, int(limit)),
                ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 是可选加速层，查询异常不得中断长期记忆召回。
            self._fts_available = False
            return ()
        return tuple(int(row[0]) for row in rows)

    def _fetch_memories_by_ids(
        self,
        memory_ids: Sequence[int],
        *,
        memory_type: str | None,
        tags: Sequence[str] | None,
    ) -> tuple[Memory, ...]:
        """分块读取索引候选，避免 SQLite 变量数量上限。"""

        identities = tuple(dict.fromkeys(int(item) for item in memory_ids))
        if not identities:
            return ()
        result: list[Memory] = []
        for offset in range(0, len(identities), 400):
            chunk = identities[offset : offset + 400]
            conditions = [f"id IN ({','.join('?' for _ in chunk)})"]
            params: list[object] = list(chunk)
            if memory_type:
                conditions.append("memory_type = ?")
                params.append(self._safe_memory_type(memory_type))
            filter_tags = tuple(tags or ())
            if filter_tags:
                conditions.append(
                    "("
                    + " OR ".join(
                        "EXISTS (SELECT 1 FROM json_each(CASE WHEN json_valid(memories.tags) "
                        "THEN memories.tags ELSE '[]' END) "
                        "WHERE json_each.type = 'text' AND json_each.value = ?)"
                        for _ in filter_tags
                    )
                    + ")"
                )
                params.extend(str(tag) for tag in filter_tags)
            with self._database._lock:
                rows = self._database.connection.execute(
                    "SELECT * FROM memories WHERE " + " AND ".join(conditions),
                    params,
                ).fetchall()
            result.extend(self._from_row(row) for row in rows)
        return tuple(result)

    def _high_priority_memory_ids(
        self,
        *,
        memory_type: str | None,
        tags: Sequence[str] | None,
    ) -> tuple[int, ...]:
        """读取所有达到必召回阈值的候选编号，不受普通候选窗口截断。"""

        threshold = int(self._settings.always_recall_priority)
        conditions = ["importance >= ?"]
        parameters: list[object] = [threshold]
        if memory_type:
            conditions.append("memory_type = ?")
            parameters.append(self._safe_memory_type(memory_type))
        filter_tags = tuple(tags or ())
        if filter_tags:
            conditions.append(
                "("
                + " OR ".join(
                    "EXISTS (SELECT 1 FROM json_each(CASE WHEN json_valid(memories.tags) "
                    "THEN memories.tags ELSE '[]' END) "
                    "WHERE json_each.type = 'text' AND json_each.value = ?)"
                    for _ in filter_tags
                )
                + ")"
            )
            parameters.extend(str(tag) for tag in filter_tags)
        # max_memories 是数据库硬上限，结果有界且足以覆盖每一条合法的高优先级记忆。
        parameters.append(self._settings.max_memories)
        with self._database._lock:
            rows = self._database.connection.execute(
                "SELECT id FROM memories WHERE "
                + " AND ".join(conditions)
                + " ORDER BY importance DESC, updated DESC, id DESC LIMIT ?",
                parameters,
            ).fetchall()
        return tuple(int(row["id"]) for row in rows)

    def search(
        self,
        query: str,
        *,
        limit: int = 7,
        memory_type: str | None = None,
        tags: Sequence[str] | None = None,
        _use_semantic: bool = True,
        _mark_recalled: bool = True,
        _cancel_event: threading.Event | None = None,
    ) -> tuple[Memory, ...]:
        """检索记忆并按相关性与持久化优先级排序。

        ``importance``（通过 ``Memory.priority`` 暴露）参与正式排序，而不是只
        用于极小的并列打破项；这样高优先级事实在语义相近时稳定排在前面。
        """

        safe_limit = max(1, min(int(limit), min(100, self._settings.max_memories)))
        query_text = str(query or "").strip()
        if _cancel_event is not None and _cancel_event.is_set():
            return ()
        query_embedding = _compute_embedding(query_text)
        semantic_scores: dict[int, float] = {}
        if query_text:
            indexed_limit = min(
                self._settings.max_memories,
                max(64, safe_limit * 8),
            )
            indexed_ids = tuple(
                memory_id
                for memory_id, _ in self._vector_index.search(
                    query_embedding,
                    limit=indexed_limit,
                )
            )
            semantic_results = (
                self._semantic.search(
                    query_text,
                    limit=indexed_limit,
                    cancel_event=_cancel_event,
                )
                if _use_semantic
                else ()
            )
            if _cancel_event is not None and _cancel_event.is_set():
                return ()
            semantic_scores = {memory_id: score for memory_id, score in semantic_results}
            semantic_ids = tuple(memory_id for memory_id, _ in semantic_results)
            lexical_ids = self._fts_candidate_ids(query_text, min(indexed_limit, 200))
            priority_memories = self.list(
                limit=min(self._settings.max_memories, max(16, safe_limit * 2)),
                memory_type=memory_type,
                tags=tags,
            )
            memories_by_id = {item.id: item for item in priority_memories}
            for item in self._fetch_memories_by_ids(
                self._high_priority_memory_ids(memory_type=memory_type, tags=tags),
                memory_type=memory_type,
                tags=tags,
            ):
                memories_by_id[item.id] = item
            for item in self._fetch_memories_by_ids(
                (*semantic_ids, *indexed_ids, *lexical_ids),
                memory_type=memory_type,
                tags=tags,
            ):
                memories_by_id[item.id] = item
            memories = tuple(memories_by_id.values())
        else:
            memories = self.list(
                limit=self._settings.max_memories,
                memory_type=memory_type,
                tags=tags,
            )
        if not memories:
            return ()
        if not query_text:
            selected = tuple(item for item in memories if self._memory_is_active(item))[:safe_limit]
            if _mark_recalled and not (_cancel_event is not None and _cancel_event.is_set()):
                self._mark_recalled(memory.id for memory in selected)
            return selected

        now = time.time()
        scored: list[tuple[float, Memory]] = []
        query_folded = query_text.casefold()
        for memory in memories:
            if not self._memory_is_active(memory):
                continue
            # 只有真实稠密模型的 ANN 命中才使用语义分数；其余记录继续使用
            # 确定性稀疏词法相似度，诊断会明确报告当前降级层。
            similarity = semantic_scores.get(
                memory.id,
                _cosine_similarity_sparse(query_embedding, memory.embedding),
            )
            exact = 1.0 if query_folded in memory.content.casefold() else 0.0
            tag_match = 1.0 if any(query_folded in tag.casefold() for tag in memory.tags) else 0.0
            if (
                similarity < self._settings.recall_min_similarity
                and exact == 0.0
                and tag_match == 0.0
                and memory.importance < self._settings.always_recall_priority
            ):
                continue
            score = self._recall_score(
                memory,
                similarity=similarity,
                exact=exact,
                tag_match=tag_match,
                now=now,
                weights=self._settings.recall_weights,
                calibration=self.recall_calibration(),
            )
            scored.append((score, memory))
        scored.sort(
            key=lambda item: (
                item[0],
                item[1].importance,
                _safe_float(item[1].last_recalled),
                _safe_float(item[1].created),
                item[1].id,
            ),
            reverse=True,
        )
        selected = tuple(memory for _, memory in scored[:safe_limit])
        if _mark_recalled and not (_cancel_event is not None and _cancel_event.is_set()):
            self._mark_recalled(memory.id for memory in selected)
        # 召回质量快照：命中按“相关性/精确/标签任一非零”判定，只记录
        # 首项分；单调递增、有界窗口，失败关闭不进 SQLite。
        self._record_recall_sample(scored, selected, similarity if query_text else 0.0)
        return selected

    def _record_recall_sample(
        self,
        scored: Sequence[tuple[float, Memory]],
        selected: Sequence[Memory],
        query_similarity: float,
    ) -> None:
        """把本次召回的快照追加到有环窗口，供状态层计算命中率。"""

        del query_similarity  # 保留参数位兼容扩展；命中只看返回集。
        hit = bool(selected) and any(score > 0.0 for score, _ in scored[:1])
        top = scored[0][0] if scored else 0.0
        with self._recall_stats_lock:
            if len(self._recall_stats_window) >= self._recall_stats_max_entries:
                self._recall_stats_window.pop(0)
            self._recall_stats_window.append((hit, float(top)))

    def search_memories(
        self,
        query: str,
        limit: int = 7,
        memory_type: str | None = None,
        tags: Sequence[str] | None = None,
        *,
        _use_semantic: bool = True,
        _mark_recalled: bool = True,
        _cancel_event: threading.Event | None = None,
    ) -> list[dict[str, Any]]:
        """兼容旧门面，返回解析后的字典而不是 SQLite 行对象。"""

        memories = self.search(
            query,
            limit=limit,
            memory_type=memory_type,
            tags=tags,
            _use_semantic=_use_semantic,
            _mark_recalled=_mark_recalled,
            _cancel_event=_cancel_event,
        )
        return [self._memory_to_dict(item) for item in memories]

    def record_event(
        self,
        event_type: str,
        description: str,
        data: Mapping[str, Any] | None = None,
    ) -> int:
        payload = self._safe_metadata(data)
        with self._database.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events (event_type, description, data, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(event_type or "system").strip() or "system",
                    str(description or "").strip(),
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )
        lastrowid = cursor.lastrowid
        return int(lastrowid) if lastrowid is not None else 0

    def add_event(
        self,
        event_type: str,
        description: str,
        data: Mapping[str, Any] | None = None,
    ) -> int:
        """旧 API 别名。"""

        return self.record_event(event_type, description, data)

    def events(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        safe_limit = max(1, min(int(limit), 500))
        with self._database._lock:
            rows = self._database.connection.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        result = []
        for row in rows:
            data = _json_load(row["data"], {})
            result.append(
                {
                    "id": int(row["id"]),
                    "event_type": str(row["event_type"]),
                    "description": str(row["description"]),
                    "data": data if isinstance(data, dict) else {},
                    "timestamp": float(row["timestamp"]),
                }
            )
        return tuple(result)

    def get_recent_events(self, limit: int = 10) -> list[dict[str, Any]]:
        """旧 API 别名，保持数据库中的倒序事件顺序。"""

        return list(self.events(limit=limit))

    def _mark_recalled(self, memories: Sequence[Memory] | Any) -> None:
        ids = [int(item.id if isinstance(item, Memory) else item) for item in memories]
        if not ids:
            return
        now = time.time()
        with self._database.transaction() as connection:
            connection.executemany(
                "UPDATE memories SET last_recalled = ?, "
                "access_count = access_count + 1 WHERE id = ?",
                ((now, memory_id) for memory_id in ids),
            )

    def _enforce_cap(
        self,
        connection: Any,
        *,
        protected_ids: Sequence[int] = (),
    ) -> tuple[int, ...]:
        """严格执行总容量，并保证当前事务刚创建的记录不会变成幽灵结果。"""

        count = int(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        excess = count - self._settings.max_memories
        if excess <= 0:
            return ()
        protected = tuple(dict.fromkeys(int(memory_id) for memory_id in protected_ids))
        exclusion = ""
        parameters: tuple[object, ...]
        if protected:
            placeholders = ",".join("?" for _ in protected)
            exclusion = f"WHERE id NOT IN ({placeholders})"
            parameters = (*protected, excess)
        else:
            parameters = (excess,)
        rows = connection.execute(
            f"""
            SELECT id FROM memories
            {exclusion}
            ORDER BY
                CASE memory_type
                    WHEN 'milestone' THEN 2
                    WHEN 'summary' THEN 1
                    ELSE 0
                END ASC,
                importance ASC,
                COALESCE(last_recalled, created) ASC,
                created ASC,
                id ASC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        removed_ids = tuple(int(row["id"]) for row in rows)
        connection.executemany(
            "DELETE FROM memories WHERE id = ?",
            ((memory_id,) for memory_id in removed_ids),
        )
        return removed_ids

    def _enforce_memory_cap(
        self,
        connection: Any | None = None,
        avoid_type: Sequence[str] = ("milestone",),
    ) -> int:
        """兼容旧维护入口；所有类型都必须服从统一的硬容量上限。"""

        del avoid_type
        if connection is None:
            with self._database.transaction() as transaction:
                removed_ids = self._enforce_cap(transaction)
        else:
            removed_ids = self._enforce_cap(connection)
        self._vector_index.remove_many(removed_ids)
        if removed_ids:
            self._semantic.notify_mutation()
        return len(removed_ids)

    def create_memory(
        self,
        content: str,
        importance: int = 1,
        memory_type: str = "fact",
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        source_ids: Sequence[int] | None = None,
        decay_factor: float = 1.0,
        source: str = "",
        priority: int | None = None,
        **kwargs: Any,
    ) -> int:
        """兼容旧 API，返回新记忆的整数 ID。"""

        # 新门面允许显式传 ``source``，未知关键字必须拒绝，避免静默丢失数据。
        if kwargs:
            unknown = ", ".join(sorted(str(key) for key in kwargs))
            raise TypeError(f"unsupported memory fields: {unknown}")
        return self.create(
            content,
            importance=importance,
            source=source,
            tags=() if tags is None else tags,
            metadata=metadata,
            memory_type=memory_type,
            source_ids=() if source_ids is None else source_ids,
            decay_factor=decay_factor,
            priority=priority,
        ).id

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT * FROM memories WHERE id = ?", (int(memory_id),)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def update_memory(self, memory_id: int, **kwargs: Any) -> bool:
        allowed = {
            "content",
            "importance",
            "priority",
            "source",
            "tags",
            "metadata",
            "memory_type",
            "source_ids",
            "decay_factor",
        }
        updates = {key: value for key, value in kwargs.items() if key in allowed}
        if not updates:
            return False
        return self.update(memory_id, **updates) is not None

    def delete_memory(self, memory_id: int) -> bool:
        return self.delete(memory_id)

    def list_memories(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created",
        memory_type: str | None = None,
        tags: Sequence[str] | None = None,
        include_archived: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        del include_archived
        safe_page = max(1, int(page))
        safe_size = max(1, min(int(page_size), 500))
        allowed_sort = {"created", "updated", "importance", "priority", "last_recalled"}
        order = (
            "importance"
            if sort_by == "priority"
            else (sort_by if sort_by in allowed_sort else "created")
        )
        conditions: list[str] = []
        params: list[object] = []
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(str(memory_type))
        if tags:
            values = self._safe_tags(tags)
            conditions.append("(" + " OR ".join("tags LIKE ?" for _ in values) + ")")
            params.extend(f'%"{value}"%' for value in values)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._database._lock:
            connection = self._database.connection
            total = int(
                connection.execute(f"SELECT COUNT(*) FROM memories{where}", params).fetchone()[0]
            )
            rows = connection.execute(
                f"SELECT * FROM memories{where} ORDER BY {order} DESC LIMIT ? OFFSET ?",
                (*params, safe_size, (safe_page - 1) * safe_size),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows], total

    def get_memories_by_tag(self, tag: str) -> list[dict[str, Any]]:
        rows = self.list(tag=str(tag), limit=500)
        return [self._memory_to_dict(item) for item in rows]

    def add_memory(
        self,
        content: str,
        importance: int = 1,
        source: str = "",
        *,
        priority: int | None = None,
    ) -> int:
        return self.create_memory(content, importance=importance, source=source, priority=priority)

    def get_important_memories(self, limit: int = 10) -> list[str]:
        return [item.content for item in self.list(limit=max(1, min(int(limit), 500)))]

    def set_master_info(self, key: str, value: str) -> None:
        safe_key = str(key or "").strip()
        safe_value = str(value or "")
        if not safe_key or len(safe_key) > 256 or len(safe_value) > MAX_TEXT_LENGTH:
            raise ValueError("master info is invalid")
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO master_info (key, value, updated)
                VALUES (?, ?, ?)
                """,
                (safe_key, safe_value, time.time()),
            )

    def get_master_info(self, key: str) -> str | None:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT value FROM master_info WHERE key = ?", (str(key),)
            ).fetchone()
        return str(row["value"]) if row else None

    def get_all_master_info(self) -> dict[str, str]:
        with self._database._lock:
            rows = self._database.connection.execute(
                "SELECT key, value FROM master_info"
            ).fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def add_chat(self, role: str, content: str, mood: str = "neutral") -> int:
        text = str(content or "").strip()
        if not text or len(text) > MAX_TEXT_LENGTH:
            raise ValueError("chat content is invalid")
        role_value = str(role or "user").strip() or "user"
        now = time.time()
        with self._database.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO chat_history (role, content, mood, timestamp, summarized)
                VALUES (?, ?, ?, ?, 0)
                """,
                (role_value, text, str(mood or "neutral"), now),
            )
            connection.execute(
                "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                ("last_chat", str(now)),
            )
            if role_value in {"mea", "assistant"}:
                row = connection.execute(
                    "SELECT value FROM mea_state WHERE key = ?", ("total_chats",)
                ).fetchone()
                try:
                    total = int(row["value"]) if row else 0
                except (TypeError, ValueError):
                    total = 0
                connection.execute(
                    "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                    ("total_chats", str(max(0, total) + 1)),
                )
            first_row = connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", ("first_met",)
            ).fetchone()
            try:
                first_met = float(first_row["value"]) if first_row else now
            except (TypeError, ValueError):
                first_met = now
            elapsed_days = max(0, int((now - first_met) / 86400) + 1)
            days_row = connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", ("total_days",)
            ).fetchone()
            try:
                old_days = int(days_row["value"]) if days_row else 0
            except (TypeError, ValueError):
                old_days = 0
            if elapsed_days > old_days:
                connection.execute(
                    "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                    ("total_days", str(elapsed_days)),
                )
        lastrowid = cursor.lastrowid
        return int(lastrowid) if lastrowid is not None else 0

    def get_recent_chats(
        self,
        limit: int = 20,
        exclude_summarized: bool = True,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        condition = " WHERE summarized = 0" if exclude_summarized else ""
        with self._database._lock:
            rows = self._database.connection.execute(
                f"SELECT * FROM chat_history{condition} ORDER BY id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_recent_chat_count(self, hours: float = 24) -> int:
        cutoff = time.time() - max(0.0, float(hours)) * 3600
        with self._database._lock:
            return int(
                self._database.connection.execute(
                    "SELECT COUNT(*) FROM chat_history WHERE timestamp >= ?", (cutoff,)
                ).fetchone()[0]
            )

    def store_chat_exchange(self, user_msg: str, mea_reply: str) -> int | None:
        if not self.enabled:
            return None
        if len(str(user_msg or "")) + len(str(mea_reply or "")) < self._settings.exchange_min_chars:
            return None
        return self.create_memory(
            f"主人：{str(user_msg)[:EXCHANGE_TRUNCATE]}\n梅尔：{str(mea_reply)[:EXCHANGE_TRUNCATE]}",
            importance=self._settings.exchange_importance,
            memory_type="exchange",
            tags=("conversation",),
            decay_factor=self._settings.exchange_decay_factor,
        )

    def get_recent_exchanges(self, limit: int = 10) -> list[str]:
        safe_limit = max(1, min(int(limit), 500))
        with self._database._lock:
            rows = self._database.connection.execute(
                """
                SELECT content FROM memories
                WHERE memory_type = 'exchange'
                ORDER BY id DESC LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [str(row["content"]) for row in reversed(rows)]

    def record_chat_event(self, role: str, content: str) -> int:
        return self.add_chat(role, content)

    def get_last_chat_time(self) -> float:
        return self._state_float("last_chat", 0.0)

    def get_first_met(self) -> float:
        return self._state_float("first_met", time.time())

    def get_master_name(self) -> str:
        return self._state_value("master_name", "主人")

    def get_nickname(self) -> str:
        return self._state_value("nickname", "")

    def get_total_chats(self) -> int:
        return self._state_int("total_chats", 0)

    def get_total_days(self) -> int:
        return self._state_int("total_days", 0)

    def mark_today_chatted(self) -> int:
        day_key = f"chatted_{datetime.now().date().isoformat()}"
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", (day_key,)
            ).fetchone()
            try:
                count = int(row["value"]) if row else 0
            except (TypeError, ValueError):
                count = 0
            count = max(0, count) + 1
            connection.execute(
                "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                (day_key, str(count)),
            )
        return count

    def get_today_chat_count(self) -> int:
        key = f"chatted_{datetime.now().date().isoformat()}"
        return self._state_int(key, 0)

    def lifecycle_maintenance(self, now: float | None = None) -> dict[str, int]:
        if not self.enabled:
            return {"decayed": 0, "pruned": 0}
        current_time = float(now if now is not None else time.time())
        decayed = 0
        removed_ids: list[int] = []
        with self._database.transaction() as connection:
            rows = connection.execute("SELECT * FROM memories").fetchall()
            for row in rows:
                points = (
                    _safe_float(row["last_decay"]),
                    _safe_float(row["last_recalled"]),
                    _safe_float(row["created"]),
                )
                last_point = max(points) or current_time
                elapsed_days = (current_time - last_point) / 86400
                if (
                    elapsed_days < self._settings.decay_days
                    or _safe_float(row["decay_factor"]) <= 0
                ):
                    continue
                steps = max(1, int(elapsed_days // self._settings.decay_days))
                floor = DECAY_FLOOR_SUMMARY if row["memory_type"] == "summary" else DECAY_FLOOR_FACT
                importance = max(floor, _safe_int(row["importance"], 1) - steps)
                connection.execute(
                    "UPDATE memories SET importance = ?, last_decay = ?, updated = ? WHERE id = ?",
                    (importance, current_time, current_time, row["id"]),
                )
                decayed += 1
            stale_rows = connection.execute(
                """
                SELECT id FROM memories
                WHERE importance <= ?
                  AND memory_type NOT IN ('summary', 'milestone')
                  AND (
                    last_recalled IS NULL
                    OR ? - last_recalled >= ?
                  )
                """,
                (
                    self._settings.prune_importance_floor,
                    current_time,
                    self._settings.prune_days * 86400,
                ),
            ).fetchall()
            stale_ids = [int(row["id"]) for row in stale_rows]
            connection.executemany(
                "DELETE FROM memories WHERE id = ?",
                ((memory_id,) for memory_id in stale_ids),
            )
            removed_ids.extend(stale_ids)
            removed_ids.extend(self._enforce_cap(connection))
        self._vector_index.remove_many(removed_ids)
        if removed_ids:
            self._semantic.notify_mutation()
        # 维护在独立事务外执行，避免长事务持有写锁；合并会重新计算保留记录的字段。
        pruned = len(removed_ids) + self._remove_exact_duplicates()
        self.consolidate_memories()
        return {"decayed": decayed, "pruned": pruned}

    def _remove_exact_duplicates(self) -> int:
        with self._database.transaction() as connection:
            # 相同内容去重时先保留优先级更高、更新更近的记录，避免
            # 维护过程把重要记忆误删后只留下低优先级副本。
            rows = connection.execute(
                "SELECT id, content FROM memories ORDER BY importance DESC, updated DESC, id DESC"
            ).fetchall()
            seen: set[str] = set()
            duplicate_ids: list[int] = []
            for row in rows:
                digest = _content_hash(row["content"])
                if digest in seen:
                    duplicate_ids.append(int(row["id"]))
                else:
                    seen.add(digest)
            for memory_id in duplicate_ids:
                connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._vector_index.remove_many(duplicate_ids)
        if duplicate_ids:
            self._semantic.notify_mutation()
        return len(duplicate_ids)

    def daily_maintenance(self) -> dict[str, int]:
        if not self.enabled:
            return {"decayed": 0, "pruned": 0}
        day_key = f"chatted_{datetime.now().date().isoformat()}"
        if self._state_value(day_key, "") == "":
            self._set_state(day_key, "0")
        return self.lifecycle_maintenance()

    def _unsummarized_stats(self) -> tuple[int, float]:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT COUNT(*) AS count, MIN(timestamp) AS oldest "
                "FROM chat_history WHERE summarized = 0"
            ).fetchone()
        if row is None:
            return 0, 0.0
        return max(0, _safe_int(row["count"])), _safe_float(row["oldest"])

    def summarization_reason(self, now: float | None = None) -> str | None:
        """按消息阈值、跨日、固定间隔的优先顺序返回触发原因。"""

        settings = self._settings
        if not self.enabled or not settings.summarization_enabled:
            return None
        count, oldest = self._unsummarized_stats()
        if count < settings.summary_min_messages or oldest <= 0:
            return None
        if self._state_int("messages_since_summary", 0) >= settings.summarize_every_n:
            return "message_count"
        current = _safe_float(now, time.time()) if now is not None else time.time()
        if settings.summarize_daily and (
            datetime.fromtimestamp(oldest).date() < datetime.fromtimestamp(current).date()
        ):
            return "daily"
        if settings.summary_interval_minutes > 0:
            last_summary = self._state_float("last_summary_at", 0.0)
            interval_anchor = max(oldest, last_summary)
            if current - interval_anchor >= settings.summary_interval_minutes * 60:
                return "interval"
        return None

    def check_summarization_trigger(self, now: float | None = None) -> bool:
        """兼容旧布尔接口；详细原因由 ``summarization_reason`` 提供。"""

        return self.summarization_reason(now) is not None

    def increment_message_counter(self) -> int:
        if not self.enabled:
            return 0
        current = self._state_int("messages_since_summary", 0) + 1
        self._set_state("messages_since_summary", str(current))
        return current

    def reset_summarization_counter(self) -> None:
        self._set_state("messages_since_summary", "0")

    def prepare_summarization_context(
        self,
        limit: int | None = None,
    ) -> tuple[list[dict[str, str]] | None, list[int] | None]:
        if not self.enabled:
            return None, None
        selected_limit = self._settings.summary_chat_limit if limit is None else int(limit)
        chats = self.get_recent_chats(
            min(selected_limit, self._settings.summary_chat_limit), exclude_summarized=True
        )
        if len(chats) < self._settings.summary_min_messages:
            return None, None
        return (
            [
                {
                    "role": (
                        "assistant"
                        if str(item["role"]).strip().lower() in {"assistant", "mea"}
                        else "user"
                    ),
                    "content": str(item["content"]),
                }
                for item in chats
            ],
            [int(item["id"]) for item in chats],
        )

    def prepare_summarization_batch(
        self,
        *,
        now: float | None = None,
        force: bool = False,
        limit: int | None = None,
    ) -> SummarizationBatch | None:
        """构造确定性的总结任务；不会在此处调用模型或修改聊天状态。"""

        if not self.enabled:
            return None
        current = _safe_float(now, time.time()) if now is not None else time.time()
        reason = "manual" if force else self.summarization_reason(current)
        if reason is None:
            return None
        messages, source_ids = self.prepare_summarization_context(limit)
        if not messages or not source_ids:
            return None
        return SummarizationBatch(
            reason=reason,
            messages=tuple(dict(item) for item in messages),
            source_ids=tuple(source_ids),
            created_at=current,
        )

    def store_summary(
        self,
        summary_text: str,
        source_ids: Sequence[int],
        importance: int = 3,
        *,
        trigger_reason: str | None = None,
        summarized_at: float | None = None,
    ) -> int:
        if not self.enabled:
            return 0
        text = str(summary_text or "").strip()
        if not text or len(text) > MAX_TEXT_LENGTH:
            raise ValueError("memory content must contain 1 to 10000 characters")
        safe_source_ids = self._safe_source_ids(source_ids)
        if not safe_source_ids:
            return 0
        placeholders = ",".join("?" for _ in safe_source_ids)
        current = (
            _safe_float(summarized_at, time.time()) if summarized_at is not None else time.time()
        )
        raw_reason = str(trigger_reason or self.summarization_reason(current) or "manual").strip()
        reason = (
            raw_reason
            if raw_reason in {"message_count", "daily", "interval", "manual"}
            else "manual"
        )
        normalized_importance = _normalize_priority(importance, "importance")
        removed_ids: tuple[int, ...] = ()
        with self._database.transaction() as connection:
            rows = connection.execute(
                f"SELECT id, role FROM chat_history "
                f"WHERE summarized = 0 AND id IN ({placeholders})",
                safe_source_ids,
            ).fetchall()
            active_source_ids = tuple(int(row["id"]) for row in rows)
            if not active_source_ids:
                return 0
            created, removed_ids = self._insert_memory_unlocked(
                connection,
                content=text,
                importance=normalized_importance,
                source="conversation:auto_summary",
                tags=("auto-summary", f"summary:{reason}"),
                metadata={
                    "summary_reason": reason,
                    "summary_message_count": len(active_source_ids),
                    "summarized_at": current,
                },
                memory_type="summary",
                source_ids=active_source_ids,
                decay_factor=0.5,
                now=current,
            )
            connection.executemany(
                "UPDATE chat_history SET summarized = 1 WHERE id = ?",
                ((source_id,) for source_id in active_source_ids),
            )
            summarized_turns = sum(1 for row in rows if str(row["role"]) == "user")
            remaining_turns = max(
                0,
                self._state_int("messages_since_summary", 0) - summarized_turns,
            )
            self._set_state_unlocked("messages_since_summary", str(remaining_turns))
            self._set_state_unlocked("last_summary_at", str(current))
            self._set_state_unlocked("last_summary_reason", reason)
        self._vector_index.upsert(created.id, created.embedding)
        self._vector_index.remove_many(removed_ids)
        self._semantic.notify_mutation()
        return created.id

    def _state_int(self, key: str, default: int) -> int:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", (key,)
            ).fetchone()
        try:
            return int(row["value"]) if row else default
        except (TypeError, ValueError):
            return default

    def _set_state(self, key: str, value: str) -> None:
        with self._database.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                (str(key), str(value)),
            )

    def _set_state_unlocked(self, key: str, value: str) -> None:
        self._database.connection.execute(
            "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
            (str(key), str(value)),
        )

    def _state_value(self, key: str, default: str = "") -> str:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", (str(key),)
            ).fetchone()
        return str(row["value"]) if row and row["value"] is not None else default

    def _get_state(self, key: str, default: str = "") -> str:
        return self._state_value(key, default)

    def _get_tier_for(self, affection: int) -> tuple[int, str, str]:
        return self._affection_service().tier_for(affection).as_tuple()

    @staticmethod
    def _parse_emb(value: object) -> list[list[float | int]]:
        parsed = _json_load(value, [])
        if not isinstance(parsed, list):
            return []
        result: list[list[float | int]] = []
        for item in parsed:
            if _valid_embedding_item(item):
                result.append([int(item[0]), float(item[1])])
        return result

    @staticmethod
    def _parse_json_list(value: object) -> list[Any]:
        parsed = _json_load(value, [])
        return list(parsed) if isinstance(parsed, list) else []

    @staticmethod
    def _parse_json_dict(value: object) -> dict[str, Any]:
        parsed = _json_load(value, {})
        return dict(parsed) if isinstance(parsed, dict) else {}

    def _state_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self._state_value(key, str(default)))
        except (TypeError, ValueError):
            return default

    def _affection_service(self) -> Any:
        if self._affection is None:
            from services.affection.service import AffectionService

            self._affection = AffectionService(self._database)
        return self._affection

    def get_affection(self) -> int:
        return int(self._affection_service().get())

    def add_affection(self, delta: int = 1) -> str | None:
        return self._affection_service().add_affection(delta)

    def get_affection_tier(self) -> tuple[int, str, str]:
        return self._affection_service().get_affection_tier()

    def get_mood(self) -> str:
        return self._affection_service().get_mood()

    def set_mood(self, mood: str) -> str:
        return self._affection_service().set_mood(mood)

    def build_context_prompt(
        self,
        current_query: str = "",
        max_chars: int | None = None,
        *,
        _use_semantic: bool = True,
        _mark_recalled: bool = True,
        _cancel_event: threading.Event | None = None,
        _recalled_ids: list[int] | None = None,
    ) -> str:
        """生成带来源标记的有限记忆上下文，避免无限膨胀 system prompt。"""

        if _cancel_event is not None and _cancel_event.is_set():
            return ""
        if not self.enabled:
            return ""
        configured_budget = (
            self._settings.context_max_chars if max_chars is None else int(max_chars)
        )
        budget = max(512, min(configured_budget, 50_000))
        affection = self.get_affection()
        tier = self.get_affection_tier()
        lines = [
            "## 与主人的关系",
            f"- 好感度：{affection}/100（{tier[1]}）",
            f"- 关系描述：{tier[2]}",
            f"- 当前心情：{self.get_mood()}",
            f"- 相识天数：{self.get_total_days()}天，共对话{self.get_total_chats()}次",
        ]
        nickname = self.get_nickname()
        if nickname:
            lines.append(f"- 主人对你的昵称：{nickname}")
        query = str(current_query or "").strip()
        if not query:
            recent = self.get_recent_chats(2, exclude_summarized=True)
            query = " ".join(str(item["content"]) for item in recent)
        memories = (
            self.search_memories(
                query,
                limit=self._settings.recall_limit,
                _use_semantic=_use_semantic,
                _mark_recalled=_mark_recalled,
                _cancel_event=_cancel_event,
            )
            if query
            else []
        )
        if _cancel_event is not None and _cancel_event.is_set():
            return ""
        if _recalled_ids is not None:
            _recalled_ids.extend(
                int(item["id"]) for item in memories if isinstance(item, Mapping) and "id" in item
            )
        if memories:
            lines.extend(("", "## 记忆（来源：长期记忆）"))
            for item in memories:
                tags = item.get("tags") or []
                priority = _normalize_priority(
                    item.get("priority", item.get("importance", 1)), "priority"
                )
                timeframe = _memory_timeframe(item)
                timeframe_prefix = f"[{timeframe}] " if timeframe else ""
                tag_prefix = f"[{', '.join(str(tag) for tag in tags)}] " if tags else ""
                prefix = f"[优先级:{priority}] {timeframe_prefix}{tag_prefix}"
                content = str(item["content"])[: self._settings.context_memory_item_chars]
                lines.append(f"- {prefix}{content}")
        if _cancel_event is not None and _cancel_event.is_set():
            return ""
        exchanges = (
            self.get_recent_exchanges(self._settings.recent_exchange_limit)
            if self._settings.recent_exchange_limit > 0
            else []
        )
        if exchanges:
            lines.extend(("", "## 近期对话（来源：对话摘要）"))
            lines.extend(f"- {item}" for item in exchanges)
        result = "\n".join(lines)
        if _cancel_event is not None and _cancel_event.is_set():
            return ""
        return result[:budget]

    async def abuild_context_prompt(
        self,
        current_query: str = "",
        max_chars: int | None = None,
    ) -> str:
        """在可取消 daemon 线程中查询语义索引，并在超时后立即词法回退。

        不使用默认 asyncio 执行器：即使第三方替身忽略取消，迟到线程也不会
        阻塞 ``asyncio.run``/运行时关闭，且线程路径关闭记忆回写。
        """

        semantic_status = self._semantic.status()
        semantic_settings = self._settings.semantic
        with self._semantic_context_lifecycle_lock:
            if self._semantic_context_closing:
                return ""
        index_size = semantic_status.get("index_size", 0)
        if not isinstance(index_size, (int, float, str)):
            index_size = 0
        if not (
            self.enabled
            and semantic_settings.enabled
            and semantic_status.get("status") == "ready"
            and int(index_size or 0) > 0
        ):
            return self.build_context_prompt(
                current_query,
                max_chars,
                _use_semantic=False,
            )
        timeout = max(
            0.05,
            min(float(self._settings.semantic.timeout_seconds), 30.0),
        )
        if not await self._acquire_semantic_context_slot(timeout):
            with self._semantic_context_lifecycle_lock:
                if self._semantic_context_closing:
                    return ""
            return self.build_context_prompt(
                current_query,
                max_chars,
                _use_semantic=False,
            )

        loop = asyncio.get_running_loop()
        cancel_event = threading.Event()
        result: asyncio.Future[tuple[str, tuple[int, ...]]] = loop.create_future()
        with self._semantic_context_lifecycle_lock:
            self._semantic_context_cancellations.add(cancel_event)

        def deliver(
            value: str | None = None,
            recalled_ids: tuple[int, ...] = (),
            error: BaseException | None = None,
        ) -> None:
            if cancel_event.is_set() or result.done():
                return
            if error is not None:
                result.set_exception(error)
            else:
                result.set_result((str(value or ""), tuple(recalled_ids)))

        def worker() -> None:
            try:
                recalled_ids: list[int] = []
                try:
                    value = self.build_context_prompt(
                        current_query,
                        max_chars,
                        _use_semantic=True,
                        _mark_recalled=False,
                        _cancel_event=cancel_event,
                        _recalled_ids=recalled_ids,
                    )
                except BaseException as exc:
                    value = None
                    error: BaseException | None = exc
                else:
                    error = None
                if cancel_event.is_set():
                    return
                try:
                    loop.call_soon_threadsafe(deliver, value, tuple(recalled_ids), error)
                except RuntimeError:
                    # 事件循环已关闭；线程不再触碰 Future、数据库或 UI。
                    return
            finally:
                self._semantic_context_slot.release()
                with self._semantic_context_lifecycle_lock:
                    self._semantic_context_cancellations.discard(cancel_event)
                    self._semantic_context_threads.discard(threading.current_thread())

        thread = threading.Thread(
            target=worker,
            name="meapet-semantic-context-query",
            daemon=True,
        )
        with self._semantic_context_lifecycle_lock:
            if self._semantic_context_closing:
                self._semantic_context_cancellations.discard(cancel_event)
                self._semantic_context_slot.release()
                return ""
            try:
                # 在生命周期锁内完成 start 与登记，close_semantic 不会捕获
                # 一个尚未启动、随后又可能访问已关闭数据库的 Thread。
                self._semantic_context_threads.add(thread)
                thread.start()
            except BaseException:
                self._semantic_context_threads.discard(thread)
                self._semantic_context_slot.release()
                self._semantic_context_cancellations.discard(cancel_event)
                raise
        try:
            value, recalled_ids = await asyncio.wait_for(asyncio.shield(result), timeout=timeout)
            current_semantic_status = self._semantic.status()
            if (
                self._settings.semantic != semantic_settings
                or current_semantic_status.get("status") != "ready"
            ):
                cancel_event.set()
                return self.build_context_prompt(
                    current_query,
                    max_chars,
                    _use_semantic=False,
                    _mark_recalled=True,
                )
            if recalled_ids and self.enabled and not cancel_event.is_set():
                try:
                    if self._semantic.status().get("status") != "closed":
                        self._mark_recalled(recalled_ids)
                except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError):
                    pass
            return value
        except asyncio.CancelledError:
            cancel_event.set()
            raise
        except TimeoutError:
            cancel_event.set()
            return self.build_context_prompt(
                current_query,
                max_chars,
                _use_semantic=False,
                _mark_recalled=True,
            )
        except Exception:
            cancel_event.set()
            return self.build_context_prompt(
                current_query,
                max_chars,
                _use_semantic=False,
                _mark_recalled=True,
            )

    async def _acquire_semantic_context_slot(self, timeout: float) -> bool:
        """在事件循环中有界等待唯一语义查询槽，不占用默认线程池。"""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.05, min(float(timeout), 30.0))
        while True:
            with self._semantic_context_lifecycle_lock:
                if self._semantic_context_closing:
                    return False
            if self._semantic_context_slot.acquire(blocking=False):
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.01, remaining))

    def wait_for_semantic_idle(self, timeout_seconds: float = 10.0) -> bool:
        """测试和有序关闭使用的后台终态等待。"""

        return self._semantic.wait_for_idle(timeout_seconds)

    def consolidate_memories(
        self,
        similarity_threshold: float | None = None,
    ) -> int:
        """合并高相似事实，保留重要性更高的记录。"""

        if not self.enabled or not self._settings.consolidation_enabled:
            return 0
        threshold = max(
            0.0,
            min(
                1.0,
                float(
                    self._settings.consolidation_similarity
                    if similarity_threshold is None
                    else similarity_threshold
                ),
            ),
        )
        memories = list(self.list(limit=self._settings.max_consolidation_memories))
        removed = 0
        consumed: set[int] = set()
        for index, left in enumerate(memories):
            if left.id in consumed or left.memory_type in {"summary", "milestone"}:
                continue
            for right in memories[index + 1 :]:
                if right.id in consumed or right.memory_type != left.memory_type:
                    continue
                similarity = _cosine_similarity_sparse(left.embedding, right.embedding)
                if similarity < threshold or self._has_contradiction(left.content, right.content):
                    continue
                keep, discard = (left, right)
                if (right.created, right.id) > (left.created, left.id):
                    keep, discard = right, left
                merged_tags = tuple(dict.fromkeys((*keep.tags, *discard.tags)))
                merged_metadata = dict(discard.metadata)
                merged_metadata.update(keep.metadata)
                merged_source_ids = tuple(dict.fromkeys((*keep.source_ids, *discard.source_ids)))[
                    :500
                ]
                merged_access_count = max(1, keep.access_count) + max(1, discard.access_count)
                merged_last_recalled = max(keep.last_recalled, discard.last_recalled)
                now = time.time()
                with self._database.transaction() as connection:
                    connection.execute(
                        """
                        UPDATE memories
                        SET importance = ?, tags = ?, metadata = ?, access_count = ?,
                            source_ids = ?, last_recalled = ?, updated = ?
                        WHERE id = ?
                        """,
                        (
                            max(keep.importance, discard.importance),
                            json.dumps(merged_tags, ensure_ascii=False),
                            json.dumps(merged_metadata, ensure_ascii=False),
                            merged_access_count,
                            json.dumps(merged_source_ids, ensure_ascii=False),
                            merged_last_recalled,
                            now,
                            keep.id,
                        ),
                    )
                    connection.execute("DELETE FROM memories WHERE id = ?", (discard.id,))
                self._vector_index.remove(discard.id)
                self._semantic.notify_mutation()
                consumed.add(discard.id)
                removed += 1
        return removed

    @staticmethod
    def _has_contradiction(left: str, right: str) -> bool:
        left_text = str(left or "").casefold()
        right_text = str(right or "").casefold()
        return any(
            (positive in left_text and negative in right_text)
            or (positive in right_text and negative in left_text)
            for positive, negative in _CONTRADICTION_PAIRS
        )

    def merge_similar_memories(self, similarity_threshold: float = CONSOLIDATION_SIMILARITY) -> int:
        return self.consolidate_memories(similarity_threshold)

    def export_data(self) -> dict[str, Any]:
        """导出 schema v7 业务数据，不包含临时每日计数。"""

        transient_prefixes = ("affection_gained_", "chatted_")
        payload: dict[str, Any] = {
            "version": 2,
            "exported_at": time.time(),
            "memories": [],
            "master_info": self.get_all_master_info(),
            "events": [],
            "state": {},
        }
        with self._database._lock:
            connection = self._database.connection
            memory_rows = connection.execute("SELECT * FROM memories ORDER BY id").fetchall()
            event_rows = connection.execute("SELECT * FROM events ORDER BY id").fetchall()
            state_rows = connection.execute("SELECT key, value FROM mea_state").fetchall()
        payload["memories"] = [self._row_to_dict(row) for row in memory_rows]
        payload["events"] = [
            {
                "id": int(row["id"]),
                "event_type": str(row["event_type"]),
                "description": str(row["description"]),
                "data": _json_load(row["data"], {}),
                "timestamp": float(row["timestamp"]),
            }
            for row in event_rows
        ]
        payload["state"] = {
            str(row["key"]): str(row["value"])
            for row in state_rows
            if not any(str(row["key"]).startswith(prefix) for prefix in transient_prefixes)
            and str(row["key"]) != "messages_since_summary"
        }
        return payload

    def export_json(self, destination: str | Path | None = None) -> dict[str, Any]:
        payload = self.export_data()
        if destination is not None:
            path = Path(destination).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return payload

    def export_to_json(self, filepath: str | Path | None = None) -> dict[str, Any]:
        return self.export_json(filepath)

    @staticmethod
    def _load_import_payload(source: object) -> dict[str, Any]:
        if isinstance(source, Mapping):
            payload = dict(source)
            try:
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("import payload is not JSON serializable") from exc
            if len(encoded) > MAX_IMPORT_BYTES:
                raise ValueError("import payload is too large")
        elif isinstance(source, (str, Path)):
            raw_source = str(source)
            try:
                path = Path(raw_source).expanduser()
                path_exists = path.exists()
            except (OSError, ValueError):
                path = None
                path_exists = False
            if path_exists and path is not None:
                if not path.is_file() or path.stat().st_size > MAX_IMPORT_BYTES:
                    raise ValueError("import file is invalid or too large")
                raw = path.read_bytes()
            else:
                raw = raw_source.encode("utf-8")
                if len(raw) > MAX_IMPORT_BYTES:
                    raise ValueError("import payload is too large")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("import payload is not valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("import payload must be an object")
        else:
            raise TypeError("source must be a path, JSON text, or mapping")
        return payload

    def import_json(self, source: object, merge: bool = False) -> int:
        payload = self._load_import_payload(source)
        raw_memories = payload.get("memories", [])
        if not isinstance(raw_memories, list):
            return 0
        if len(raw_memories) > MAX_IMPORT_ITEMS:
            raise ValueError("too many memories in import")
        prepared_items: list[dict[str, Any]] = []
        for item in raw_memories:
            prepared = self._prepare_import_memory(item)
            if prepared is not None:
                prepared_items.append(prepared)
        empty_snapshot = not merge and not raw_memories
        if not prepared_items and not empty_snapshot:
            # 包含但全部非法的条目保持旧数据，避免格式错误的导入意外擦除
            # 长期记忆；空数组则是一个有效的替换快照，继续处理其 state、
            # master_info 和 events 字段。
            return 0
        cleared_existing = False
        with self._database.transaction() as connection:
            if not merge:
                connection.execute("DELETE FROM memories")
                self._vector_index.clear()
                cleared_existing = True
        if cleared_existing:
            self._semantic.notify_mutation()
        existing_hashes: set[str] = set()
        if merge:
            with self._database._lock:
                rows = self._database.connection.execute("SELECT content FROM memories").fetchall()
            existing_hashes = {_content_hash(row["content"]) for row in rows}
        imported = 0
        for item in prepared_items:
            content = item["content"]
            digest = _content_hash(content)
            if digest in existing_hashes:
                continue
            try:
                self.create_memory(
                    content,
                    importance=item["importance"],
                    memory_type=item["memory_type"],
                    tags=item["tags"],
                    metadata=item["metadata"],
                    source_ids=item["source_ids"],
                    decay_factor=item["decay_factor"],
                    source=item["source"],
                )
            except (TypeError, ValueError, OverflowError):
                continue
            existing_hashes.add(digest)
            imported += 1
        master_info = payload.get("master_info")
        if isinstance(master_info, Mapping):
            for key, value in master_info.items():
                if isinstance(value, str):
                    try:
                        self.set_master_info(str(key), value)
                    except ValueError:
                        continue
        preserved_state = {
            "affection",
            "mood",
            "nickname",
            "master_name",
            "total_chats",
            "total_days",
            "first_met",
        }
        state = payload.get("state")
        if isinstance(state, Mapping):
            for key, value in state.items():
                if str(key) in preserved_state:
                    self._set_state(str(key), str(value))
        raw_events = payload.get("events", [])
        if isinstance(raw_events, list):
            for item in raw_events[:MAX_IMPORT_ITEMS]:
                if not isinstance(item, Mapping):
                    continue
                event_type = str(item.get("event_type") or "system").strip() or "system"
                description = str(item.get("description") or "").strip()
                if len(event_type) > 256 or len(description) > MAX_TEXT_LENGTH:
                    continue
                data = item.get("data")
                if not isinstance(data, Mapping):
                    data = {}
                try:
                    event_data = self._safe_metadata(data)
                except ValueError:
                    continue
                timestamp = _safe_float(item.get("timestamp"), time.time())
                with self._database.transaction() as connection:
                    connection.execute(
                        "INSERT INTO events (event_type, description, data, timestamp) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            event_type,
                            description,
                            json.dumps(event_data, ensure_ascii=False),
                            timestamp,
                        ),
                    )
        return imported

    @classmethod
    def _prepare_import_memory(cls, item: object) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None
        content = str(item.get("content") or "").strip()
        if not content or len(content) > MAX_TEXT_LENGTH:
            return None
        tags = item.get("tags")
        if isinstance(tags, (str, bytes, bytearray)) or not isinstance(tags, Sequence):
            tags = ()
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        source_ids = item.get("source_ids")
        if isinstance(source_ids, (str, bytes, bytearray)) or not isinstance(source_ids, Sequence):
            source_ids = ()
        try:
            safe_tags = cls._safe_tags(tags)
            safe_metadata = cls._safe_metadata(metadata)
            safe_source_ids = cls._safe_source_ids(source_ids)
            raw_importance = item.get("importance", item.get("priority", 1))
            importance = _normalize_priority(
                raw_importance, "priority" if "importance" not in item else "importance"
            )
            raw_decay_factor = item.get("decay_factor", 1.0)
            if not isinstance(raw_decay_factor, (int, float, str)):
                return None
            decay_factor = float(raw_decay_factor)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(decay_factor) or decay_factor < 0:
            return None
        try:
            memory_type = cls._safe_memory_type(item.get("memory_type"))
            source = cls._safe_source(item.get("source"))
        except (TypeError, ValueError, OverflowError):
            return None
        return {
            "content": content,
            "importance": importance,
            "memory_type": memory_type,
            "tags": safe_tags,
            "metadata": safe_metadata,
            "source_ids": safe_source_ids,
            "decay_factor": decay_factor,
            "source": source,
        }

    def import_from_json(self, source: object, merge: bool = False) -> int:
        return self.import_json(source, merge=merge)

    def close(self) -> None:
        self.close_semantic()
        self._database.close()

    def close_semantic(self) -> None:
        """停止语义模型子进程和索引线程，但保留共享 SQLite 生命周期。"""
        with self._semantic_context_lifecycle_lock:
            self._semantic_context_closing = True
            cancellations = tuple(self._semantic_context_cancellations)
            threads = tuple(self._semantic_context_threads)
            for cancellation in cancellations:
                cancellation.set()
        self._semantic.close()
        # 真实 provider 有自己的有界关闭；这里再给 daemon 查询线程一个短暂
        # 收尾窗口，避免它们在共享 SQLite 已关闭后继续执行迟到回写。
        deadline = time.monotonic() + 1.0
        for thread in threads:
            if thread is threading.current_thread():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)

    def reset_all(self) -> None:
        with self._database.transaction() as connection:
            for table in ("chat_history", "mea_state", "events", "memories", "conversation_turns"):
                connection.execute(f"DELETE FROM {table}")
        self._vector_index.clear()
        self._semantic.notify_mutation()
        self._database._ensure_defaults()
