"""好感度、等级与情绪状态服务。"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from db.database import Database

AFFECTION_MIN = 0
AFFECTION_MAX = 100
AFFECTION_GAIN_PER_CHAT = 1
AFFECTION_DAILY_CAP = 15
MOODS = ("高兴", "平静", "困倦", "烦躁", "难过", "期待", "好奇", "生气", "孤独")
MOOD_ALIASES = {"开心": "高兴", "忧郁": "难过"}
_STALE_MOOD_HINT = (
    "距上次心情标记已超过 {hours} 小时；请根据你的人设与当前上下文决定是否更新"
    "心情：若认为情绪应随时间恢复，请在回答末尾的 <meapet> 指令里给出更合适或"
    "保持的 <mood>；不强制恢复平静。"
)


@dataclass(frozen=True)
class AffectionTier:
    """好感度等级。"""

    threshold: int
    name: str
    description: str = ""

    def as_tuple(self) -> tuple[int, str, str]:
        return self.threshold, self.name, self.description


TIERS = (
    AffectionTier(0, "陌生人", "……你是谁？别靠近我喵。"),
    AffectionTier(10, "认识", "嗯，记得你。有事快说喵。"),
    AffectionTier(30, "熟人", "又来啦。真是闲得慌喵。"),
    AffectionTier(50, "朋友", "哼，才不是特意等你的喵。"),
    AffectionTier(70, "好朋友", "……其实，和你聊天也不算太讨厌喵。"),
    AffectionTier(85, "亲密", "你来的话，我……稍微有点开心喵。"),
    AffectionTier(95, "挚友", "你是我少数不讨厌的人类喵。"),
)
AFFECTION_TIERS = tuple(item.as_tuple() for item in TIERS)


@dataclass(frozen=True)
class AffectionChange:
    """一次好感度调整的结果。"""

    previous: int
    current: int
    applied: int
    tier: AffectionTier

    @property
    def tier_changed(self) -> bool:
        return self.tier != AffectionService.tier_for(self.previous)


class AffectionService:
    """在事务内执行边界裁剪与每日正向增量限制。"""

    def __init__(
        self,
        database: Database,
        today: Callable[[], date] | None = None,
        *,
        decay_enabled: bool = True,
        decay_after_seconds: float = 14_400.0,
        decay_prompt_probability: float = 0.5,
    ) -> None:
        if not isinstance(database, Database):
            raise TypeError("database must be a Database")
        if not isinstance(decay_enabled, bool):
            raise ValueError("decay_enabled must be a boolean")
        if (
            isinstance(decay_after_seconds, bool)
            or not math.isfinite(float(decay_after_seconds))
            or float(decay_after_seconds) < 600.0
        ):
            raise ValueError("decay_after_seconds must be a finite number >= 600")
        if (
            isinstance(decay_prompt_probability, bool)
            or not math.isfinite(float(decay_prompt_probability))
            or not 0.0 <= float(decay_prompt_probability) <= 1.0
        ):
            raise ValueError("decay_prompt_probability must be within [0, 1]")
        self._database = database
        self._today = today or date.today
        self._decay_enabled = bool(decay_enabled)
        self._decay_after_seconds = float(decay_after_seconds)
        self._decay_prompt_probability = float(decay_prompt_probability)

    def _mood_configuration(self) -> tuple[bool, float, float]:
        """返回 (decay_enabled, decay_after_seconds, decay_prompt_probability)。"""

        return (
            self._decay_enabled,
            self._decay_after_seconds,
            self._decay_prompt_probability,
        )

    def configure_mood(
        self,
        *,
        decay_enabled: bool,
        decay_after_seconds: float,
        decay_prompt_probability: float,
    ) -> None:
        if not isinstance(decay_enabled, bool):
            raise ValueError("decay_enabled must be a boolean")
        if (
            isinstance(decay_after_seconds, bool)
            or not math.isfinite(float(decay_after_seconds))
            or float(decay_after_seconds) < 600.0
        ):
            raise ValueError("decay_after_seconds must be a finite number >= 600")
        if (
            isinstance(decay_prompt_probability, bool)
            or not math.isfinite(float(decay_prompt_probability))
            or not 0.0 <= float(decay_prompt_probability) <= 1.0
        ):
            raise ValueError("decay_prompt_probability must be within [0, 1]")
        self._decay_enabled = bool(decay_enabled)
        self._decay_after_seconds = float(decay_after_seconds)
        self._decay_prompt_probability = float(decay_prompt_probability)

    @staticmethod
    def tier_for(value: int) -> AffectionTier:
        bounded = max(AFFECTION_MIN, min(AFFECTION_MAX, int(value)))
        return max(
            (tier for tier in TIERS if bounded >= tier.threshold),
            key=lambda item: item.threshold,
        )

    @staticmethod
    def _get_tier_for(value: int) -> tuple[int, str, str]:
        return AffectionService.tier_for(value).as_tuple()

    @staticmethod
    def _read_int(row: object, default: int) -> int:
        if row is None:
            return default
        try:
            value = row["value"]  # type: ignore[index]
        except (KeyError, TypeError, IndexError):
            value = default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def get(self) -> int:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", ("affection",)
            ).fetchone()
        return max(AFFECTION_MIN, min(AFFECTION_MAX, self._read_int(row, 5)))

    def adjust(self, delta: int) -> AffectionChange:
        amount = int(delta)
        day_key = f"affection_gained_{self._today().isoformat()}"
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", ("affection",)
            ).fetchone()
            previous = max(AFFECTION_MIN, min(AFFECTION_MAX, self._read_int(row, 5)))
            gained_row = connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", (day_key,)
            ).fetchone()
            gained = max(0, self._read_int(gained_row, 0))
            allowed = min(amount, max(0, AFFECTION_DAILY_CAP - gained)) if amount > 0 else amount
            current = max(AFFECTION_MIN, min(AFFECTION_MAX, previous + allowed))
            applied = current - previous
            connection.execute(
                "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                ("affection", str(current)),
            )
            if applied > 0:
                connection.execute(
                    "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                    (day_key, str(gained + applied)),
                )
            before_tier = self.tier_for(previous)
            after_tier = self.tier_for(current)
            if before_tier != after_tier:
                connection.execute(
                    """
                    INSERT INTO events (event_type, description, data, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        "milestone",
                        f"好感度升级：{after_tier.name}（{previous}→{current}）",
                        json.dumps(
                            {
                                "previous": previous,
                                "current": current,
                                "tier": after_tier.name,
                            },
                            ensure_ascii=False,
                        ),
                        time.time(),
                    ),
                )
        return AffectionChange(previous, current, applied, after_tier)

    def set(self, value: int) -> AffectionChange:
        """直接设置值，仅供导入/管理操作，不消耗每日增量额度。"""

        bounded = max(AFFECTION_MIN, min(AFFECTION_MAX, int(value)))
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", ("affection",)
            ).fetchone()
            previous = max(AFFECTION_MIN, min(AFFECTION_MAX, self._read_int(row, 5)))
            connection.execute(
                "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                ("affection", str(bounded)),
            )
        return AffectionChange(previous, bounded, bounded - previous, self.tier_for(bounded))

    def get_affection(self) -> int:
        """兼容旧业务门面。"""

        return self.get()

    def add_affection(self, delta: int = AFFECTION_GAIN_PER_CHAT) -> str | None:
        """兼容旧门面，跨档时返回等级描述，否则返回 ``None``。"""

        change = self.adjust(delta)
        if change.tier_changed and change.current > change.previous:
            return change.tier.description
        return None

    def get_affection_tier(self) -> tuple[int, str, str]:
        tier = self.tier_for(self.get())
        return tier.as_tuple()

    def get_mood(self) -> str:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", ("mood",)
            ).fetchone()
        if row is None:
            return "平静"
        value = str(row["value"] or "平静")
        normalized = MOOD_ALIASES.get(value, value)
        return normalized if normalized in MOODS else "平静"

    def set_mood(self, mood: str) -> str:
        value = str(mood or "").strip()
        normalized = MOOD_ALIASES.get(value, value)
        if normalized not in MOODS:
            raise ValueError("mood is unsupported")
        with self._database.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                ("mood", normalized),
            )
            connection.execute(
                "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                ("mood_updated", str(time.time())),
            )
        return normalized

    def stale_mood_hint(
        self, now: float | None = None, rng: Callable[[], float] | None = None
    ) -> str:

        if not self._decay_enabled:
            return ""
        current = self.get_mood()
        if current in ("", "平静"):
            return ""
        row = self._read_mood_updated()
        if row is None:
            return ""
        try:
            updated = float(str(row))
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(updated) or updated <= 0:
            return ""
        instant = float(now if now is not None else time.time())
        if not math.isfinite(instant) or instant < updated:
            return ""
        if instant - updated < self._decay_after_seconds:
            return ""
        sampler = rng if rng is not None else random.random
        try:
            roll = float(sampler())
        except (TypeError, ValueError, OverflowError):
            return ""
        if not math.isfinite(roll) or not 0.0 <= roll <= 1.0:
            return ""
        if roll >= self._decay_prompt_probability:
            return ""
        hours = max(1, int((instant - updated) / 3600.0))
        return _STALE_MOOD_HINT.format(hours=hours)

    def _read_mood_updated(self) -> object:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", ("mood_updated",)
            ).fetchone()
        return None if row is None else row["value"]

    def get_total_chats(self) -> int:
        return self._state_int("total_chats", 0)

    def get_total_days(self) -> int:
        return self._state_int("total_days", 0)

    def mark_today_chatted(self, *, increment_total: bool = True) -> int:
        day_key = f"chatted_{self._today().isoformat()}"
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", (day_key,)
            ).fetchone()
            count = max(0, self._read_int(row, 0)) + 1
            connection.execute(
                "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                (day_key, str(count)),
            )
            if increment_total:
                total = (
                    self._read_int(
                        connection.execute(
                            "SELECT value FROM mea_state WHERE key = ?", ("total_chats",)
                        ).fetchone(),
                        0,
                    )
                    + 1
                )
                connection.execute(
                    "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
                    ("total_chats", str(total)),
                )
        return count

    def get_today_chat_count(self) -> int:
        return self._state_int(f"chatted_{self._today().isoformat()}", 0)

    def _state_int(self, key: str, default: int) -> int:
        with self._database._lock:
            row = self._database.connection.execute(
                "SELECT value FROM mea_state WHERE key = ?", (key,)
            ).fetchone()
        return max(0, self._read_int(row, default))
