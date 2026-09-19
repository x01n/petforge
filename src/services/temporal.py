from __future__ import annotations

import re
import zoneinfo
from dataclasses import dataclass
from datetime import datetime, timedelta

_UNITS = {
    "ms": 0.001,
    "s": 1.0,
    "sec": 1.0,
    "秒": 1.0,
    "m": 60.0,
    "min": 60.0,
    "分钟": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "小时": 3600.0,
}
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
_STRUCTURED_PATTERN = re.compile(
    r"\s*(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|sec|m|min|h|hr|秒|分钟|小时)\s*"
)


class TimeExpressionError(Exception):
    """时间表达无法解释或超出安全边界。"""


@dataclass(frozen=True, slots=True)
class TimePoint:
    """一天中的具体时刻，带本地时区标签。"""

    hour: int
    minute: int
    timezone: str


@dataclass(frozen=True, slots=True)
class IntervalSchedule:
    """解析后的间隔型调度表达式。"""

    seconds: float
    human: str


def parse_time_of_day(raw: object) -> TimePoint:
    """解析一天中的时刻；支持 ``HH:MM`` 与零点起算的分钟数。

    模型常用文字会先由调用方做近义改写（“晚一点”等不在这里猜测），
    这里只接受可精确计算的表达。超过一天边界或非法字段直接报错。
    """

    value = str(raw or "").strip()
    if not value:
        raise TimeExpressionError("时间表达为空")
    match = re.fullmatch(r"(0?\d|1\d|2[0-3]):([0-5]\d)", value)
    if match:
        return TimePoint(
            hour=int(match.group(1)),
            minute=int(match.group(2)),
            timezone=_default_timezone(),
        )
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        raise TimeExpressionError("时间表达不是合法时刻")
    numeric = float(value)
    if not 0 <= numeric < 1440:
        raise TimeExpressionError("时间表达超出一天范围")
    minutes = int(numeric)
    return TimePoint(
        hour=minutes // 60,
        minute=minutes % 60,
        timezone=_default_timezone(),
    )


def parse_interval(raw: object) -> IntervalSchedule:
    """解析间隔表达；只接受 ``<数量><单位>`` 与纯数字分钟格式。

    单位兼容 ``ms/s/sec/m/min/h/hr``、中文 ``秒/分钟/小时``；纯数字
    解释为分钟，便于模型写出“每5”这样简化的表达。
    """

    value = str(raw or "").strip().lower()
    if not value:
        raise TimeExpressionError("间隔表达为空")
    structured = _STRUCTURED_PATTERN.fullmatch(value)
    if structured is not None:
        amount_text = structured.group("amount")
        unit = structured.group("unit")
        if amount_text is None or unit is None:  # pragma: no cover - 模式保证存在
            raise TimeExpressionError("间隔表达无法解释")
        seconds = float(amount_text) * _UNITS[unit]
        return IntervalSchedule(seconds=seconds, human=_interval_human(seconds))
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        seconds = float(value) * 60.0
        return IntervalSchedule(seconds=seconds, human=_interval_human(seconds))
    raise TimeExpressionError("间隔表达无法解释")


def _interval_human(seconds: float) -> str:
    """把秒数转成最自然的耐受表达，避免出现过长的浮点尾巴。"""

    if seconds < 60:
        return f"{seconds:g}秒"
    minutes = seconds / 60.0
    if minutes < 120:
        return f"{minutes:g}分钟"
    hours = seconds / 3600.0
    return f"{hours:g}小时"


def parse_window_duration(raw: object) -> float:
    """解析窗口停留秒数；只接受非负有限秒数。"""

    value = float(seconds_of(raw))
    if value < 1:
        raise TimeExpressionError("停留时长至少为 1 秒")
    return value


def seconds_of(raw: object) -> str:
    """把明显的时间表达转成秒数字符串；无法解释时原样返回。"""

    text = str(raw or "").strip()
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def _default_timezone() -> str:
    """返回当前系统的 IANA 时区名；不可用时回退 UTC 保证可校验。"""

    try:
        return str(zoneinfo.ZoneInfo("localtime").key)
    except Exception as exc:  # pragma: no cover - 系统 tzdata 异常的兜底
        del exc
        return "UTC"


def next_occurrence(point: TimePoint, now: datetime | None = None) -> datetime:
    """计算给定时刻的下一次发生时间。"""

    base = now or datetime.now().astimezone()
    target = base.astimezone(zoneinfo.ZoneInfo(point.timezone))
    candidate = target.replace(hour=point.hour, minute=point.minute, second=0, microsecond=0)
    if candidate <= target:
        candidate = candidate + timedelta(days=1)
    return candidate


__all__ = [
    "TimeExpressionError",
    "TimePoint",
    "IntervalSchedule",
    "next_occurrence",
    "parse_interval",
    "parse_time_of_day",
    "parse_window_duration",
    "seconds_of",
]
