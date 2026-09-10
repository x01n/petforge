"""可配置的应用日志。

日志默认只写入标准错误，避免导入模块或无头校验时产生文件副作用。启动器在
读取 YAML 后可再次调用 :func:`configure_logging`，启用彩色控制台和轮换文件。
所有由本模块创建的处理器都会带有内部标记，重复配置时只替换自己的处理器，
不会移除宿主（例如 pytest 或 Qt）安装的处理器。
"""

from __future__ import annotations

import copy
import json
import logging
import logging.handlers
import sys
import traceback
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, TextIO

from core.util import as_mapping as _as_mapping

from .events import event_fingerprint, log_event, sanitize_log_text

__all__ = [
    "LoggingSettings",
    "MemoryLogHandler",
    "SafeFormatter",
    "ColorFormatter",
    "close_logging",
    "configure_logging",
    "event_fingerprint",
    "log_event",
    "recent_log_records",
    "clear_log_records",
    "sanitize_log_text",
]


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_HANDLER_MARKER = "_meapet_logging_handler"
_CONFIG_LOCK = RLock()
_MEMORY_HANDLER: MemoryLogHandler | None = None

_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",  # 青色
    logging.INFO: "\033[32m",  # 绿色
    logging.WARNING: "\033[33m",  # 黄色
    logging.ERROR: "\033[31m",  # 红色
    logging.CRITICAL: "\033[35m",  # 紫色
}
_RESET = "\033[0m"

# 即使应用尚未显式调用 ``configure_logging``（例如库级 IPC 测试或嵌入式
# 宿主），asyncio debug transport 也不应把非 JSON 内部诊断污染到根 sink。
logging.getLogger("asyncio").setLevel(logging.WARNING)


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "on", "1"}:
        return True
    if normalized in {"false", "no", "off", "0"}:
        return False
    raise ValueError("logging boolean value is invalid")


def _as_level(value: object, *, default: int | None = None) -> int:
    """把级别名称转换为标准 logging 数值，未知名称直接拒绝。"""

    if value is None:
        if default is not None:
            return default
        raise ValueError("logging level is required")
    if isinstance(value, bool):
        raise ValueError("logging level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    if isinstance(value, int):
        if 0 <= value <= 50:
            return value
        raise ValueError("logging level number is outside the allowed range")
    normalized = str(value).strip().upper()
    aliases = {
        "WARN": 30,
        "FATAL": 50,
    }
    if normalized in aliases:
        return aliases[normalized]
    level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(normalized)
    if level is not None:
        return level
    raise ValueError("logging level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")


@dataclass(frozen=True)
class LoggingSettings:
    """日志装配后的不可变设置。"""

    level: int = 20
    console_level: int = 20
    console_enabled: bool = True
    color: bool | None = None
    file_level: int = 20
    file_enabled: bool = False
    file_path: Path | None = None
    rotation: str = "time"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 7
    when: str = "midnight"
    interval: int = 1
    utc: bool = False
    memory_enabled: bool = True
    memory_level: int = 20
    memory_capacity: int = 1000

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None = None,
        *,
        base_directory: Path | None = None,
        level: str | int | None = None,
        log_path: str | Path | None = None,
        console: bool | None = None,
        color: bool | str | None = None,
        rotation: str | None = None,
        max_bytes: int | None = None,
        backup_count: int | None = None,
        when: str | None = None,
        interval: int | None = None,
        utc: bool | None = None,
        memory_enabled: bool | None = None,
        memory_level: str | int | None = None,
        memory_capacity: int | None = None,
    ) -> LoggingSettings:
        """从完整配置或 ``logging`` 子映射读取设置。"""

        if value is None:
            raw = {}
        elif not isinstance(value, Mapping):
            raise ValueError("logging configuration must be a mapping")
        else:
            raw = dict(value)
        if "logging" in raw:
            if not isinstance(raw["logging"], Mapping):
                raise ValueError("logging must be a mapping")
            raw = dict(raw["logging"])
        raw_console = raw.get("console")
        raw_file = raw.get("file")
        # ``console`` 只接受布尔开关或结构化配置；数值 0/1 不在这里
        # 隐式转换，避免类型错误被 ``_as_mapping`` 静默吞掉。
        if raw_console is not None and not isinstance(raw_console, (Mapping, bool)):
            raise ValueError("logging.console must be a mapping or boolean")
        if raw_file is not None and not isinstance(raw_file, (Mapping, bool, str, Path)):
            raise ValueError("logging.file must be a mapping, path, or boolean")
        console_values = _as_mapping(raw_console)
        file_values = _as_mapping(raw_file)
        raw_memory = raw.get("memory")
        if raw_memory is not None and not isinstance(raw_memory, Mapping):
            raise ValueError("logging.memory must be a mapping")
        memory_values = _as_mapping(raw_memory)

        configured_path = log_path
        if configured_path is None:
            configured_path = file_values.get("path", raw.get("log_path", raw.get("file_path")))
            if configured_path is None and isinstance(raw_file, (str, Path)):
                configured_path = raw_file
        path = None
        if configured_path is not None:
            if not isinstance(configured_path, (str, Path)):
                raise ValueError("logging.file.path must be a string")
            if not str(configured_path).strip():
                raise ValueError("logging.file.path must be a non-empty string")
            path = Path(configured_path).expanduser()
            if not path.is_absolute() and base_directory is not None:
                path = base_directory / path
            path = path.resolve()

        selected_level = _as_level(level if level is not None else raw.get("level", "INFO"))
        console_level = _as_level(
            console_values.get("level", raw.get("console_level", selected_level)),
            default=selected_level,
        )
        file_level = _as_level(
            file_values.get("level", raw.get("file_level", selected_level)),
            default=selected_level,
        )
        selected_memory_level = _as_level(
            memory_level
            if memory_level is not None
            else memory_values.get("level", raw.get("memory_level", selected_level)),
            default=selected_level,
        )
        selected_memory_enabled = _as_bool(
            memory_enabled
            if memory_enabled is not None
            else memory_values.get("enabled", raw.get("memory_enabled", True)),
            default=True,
        )
        selected_memory_capacity = (
            memory_capacity
            if memory_capacity is not None
            else memory_values.get("capacity", raw.get("memory_capacity", 1000))
        )
        if isinstance(selected_memory_capacity, bool):
            raise ValueError("logging.memory.capacity must be a positive integer")
        try:
            normalized_memory_capacity = int(selected_memory_capacity)
        except (TypeError, ValueError) as exc:
            raise ValueError("logging.memory.capacity must be a positive integer") from exc
        if not 1 <= normalized_memory_capacity <= 100_000:
            raise ValueError("logging.memory.capacity is outside the allowed range")

        console_enabled_value = (
            console
            if console is not None
            else console_values.get("enabled", raw.get("console_enabled", True))
        )
        if console is None and isinstance(raw_console, bool):
            console_enabled_value = raw_console
        console_enabled = _as_bool(console_enabled_value, default=True)

        color_value: bool | str | None = (
            color if color is not None else console_values.get("color", raw.get("color"))
        )
        if color_value is None:
            color_enabled: bool | None = None
        elif isinstance(color_value, bool):
            color_enabled = color_value
        else:
            color_key = str(color_value).strip().lower()
            if color_key in {"", "auto"}:
                color_enabled = None
            elif color_key in {"always", "true", "yes", "on", "1"}:
                color_enabled = True
            elif color_key in {"never", "false", "no", "off", "0"}:
                color_enabled = False
            else:
                raise ValueError("logging.console.color must be auto, always, or never")

        file_enabled_value = file_values.get("enabled", raw.get("file_enabled"))
        if file_enabled_value is None and isinstance(raw_file, bool):
            file_enabled_value = raw_file
        if file_enabled_value is None:
            file_enabled = path is not None
        else:
            file_enabled = _as_bool(file_enabled_value, default=False)
        if log_path is not None and file_enabled_value is None:
            file_enabled = True

        selected_rotation = rotation
        if selected_rotation is None:
            selected_rotation = file_values.get("rotation", raw.get("rotation", "time"))
        normalized_rotation = str(selected_rotation or "time").strip().lower()
        aliases = {"timed": "time", "daily": "time", "off": "none", "disabled": "none"}
        normalized_rotation = aliases.get(normalized_rotation, normalized_rotation)
        if normalized_rotation not in {"size", "time", "none"}:
            raise ValueError("logging.file.rotation must be size, time, or none")

        selected_max_bytes = (
            max_bytes
            if max_bytes is not None
            else file_values.get("max_bytes", raw.get("max_bytes", 10 * 1024 * 1024))
        )
        if isinstance(selected_max_bytes, bool):
            raise ValueError("logging.file.max_bytes must be a positive integer")
        try:
            normalized_max_bytes = int(selected_max_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("logging.file.max_bytes must be a positive integer") from exc
        if normalized_max_bytes <= 0:
            raise ValueError("logging.file.max_bytes must be a positive integer")

        selected_backup_count = (
            backup_count
            if backup_count is not None
            else file_values.get("backup_count", raw.get("backup_count", 7))
        )
        if isinstance(selected_backup_count, bool):
            raise ValueError("logging.file.backup_count must be a non-negative integer")
        try:
            normalized_backup_count = int(selected_backup_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("logging.file.backup_count must be a non-negative integer") from exc
        if normalized_backup_count < 0:
            raise ValueError("logging.file.backup_count must be a non-negative integer")

        selected_when = (
            when if when is not None else file_values.get("when", raw.get("when", "midnight"))
        )
        if not isinstance(selected_when, str) or not selected_when.strip():
            raise ValueError("logging.file.when must be a non-empty string")
        normalized_when = selected_when.strip()
        when_key = normalized_when.lower()
        if when_key not in {"s", "m", "h", "d", "midnight"} and not (
            len(normalized_when) == 2
            and normalized_when[0].upper() == "W"
            and normalized_when[1] in "0123456"
        ):
            raise ValueError("logging.file.when is invalid")
        selected_interval = (
            interval
            if interval is not None
            else file_values.get("interval", raw.get("interval", 1))
        )
        if isinstance(selected_interval, bool):
            raise ValueError("logging.file.interval must be a positive integer")
        try:
            normalized_interval = int(selected_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("logging.file.interval must be a positive integer") from exc
        if normalized_interval <= 0:
            raise ValueError("logging.file.interval must be a positive integer")

        selected_utc = utc if utc is not None else file_values.get("utc", raw.get("utc", False))
        return cls(
            level=selected_level,
            console_level=console_level,
            console_enabled=console_enabled,
            color=color_enabled,
            file_level=file_level,
            file_enabled=file_enabled,
            file_path=path,
            rotation=normalized_rotation,
            max_bytes=normalized_max_bytes,
            backup_count=normalized_backup_count,
            when=normalized_when,
            interval=normalized_interval,
            utc=_as_bool(selected_utc, default=False),
            memory_enabled=selected_memory_enabled,
            memory_level=selected_memory_level,
            memory_capacity=normalized_memory_capacity,
        )


class SafeFormatter(logging.Formatter):
    """保留异常类型和代码位置，同时移除异常正文、绝对路径和凭据。"""

    def format(self, record: logging.LogRecord) -> str:
        safe_record = copy.copy(record)
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = f"log_message_unavailable:{type(record.msg).__name__}"
        safe_record.msg = sanitize_log_text(message, limit=4096)
        safe_record.args = ()
        safe_record.exc_text = None
        return super().format(safe_record)

    def formatException(self, exc_info) -> str:
        exception_type, exception, trace = exc_info
        type_name = sanitize_log_text(
            getattr(exception_type, "__name__", "Exception"),
            limit=96,
        )
        reason_fingerprint = event_fingerprint(f"{type_name}:{exception}")
        frames = traceback.extract_tb(trace, limit=12) if trace is not None else ()
        rendered_frames = [
            (
                f"{sanitize_log_text(Path(frame.filename).name, limit=96)}:"
                f"{int(frame.lineno)}:"
                f"{sanitize_log_text(frame.name, limit=96)}"
            )
            for frame in frames
        ]
        trace_summary = "<-".join(rendered_frames) or "unavailable"
        return (
            f"exception_type={type_name} "
            f"reason_fingerprint={reason_fingerprint or 'unavailable'} "
            f"trace={trace_summary}"
        )

    def formatStack(self, stack_info: str) -> str:
        return f"stack_fingerprint={event_fingerprint(stack_info) or 'unavailable'}"


class ColorFormatter(SafeFormatter):
    """为控制台记录着色；文件格式化器始终使用纯文本。"""

    def __init__(
        self,
        fmt: str = _DEFAULT_FORMAT,
        *,
        datefmt: str = "%Y-%m-%d %H:%M:%S",
        enabled: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.enabled = bool(enabled)

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if not self.enabled:
            return rendered
        color = _LEVEL_COLORS.get(record.levelno)
        if not color:
            return rendered
        return f"{color}{rendered}{_RESET}"


class MemoryLogHandler(logging.Handler):
    """保存有界、脱敏的最近日志，供本地控制台按需查询。"""

    def __init__(self, capacity: int = 1000) -> None:
        super().__init__(level=logging.NOTSET)
        if isinstance(capacity, bool) or not 1 <= int(capacity) <= 100_000:
            raise ValueError("memory log capacity is outside the allowed range")
        self._records: deque[dict[str, object]] = deque(maxlen=int(capacity))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = sanitize_log_text(record.getMessage(), limit=4096)
            snapshot: dict[str, object] = {
                "created_at": round(float(record.created), 3),
                "level": record.levelname,
                "logger": sanitize_log_text(record.name, limit=128),
                "message": message,
            }
            try:
                structured = json.loads(message)
            except (TypeError, ValueError, json.JSONDecodeError):
                structured = None
            if isinstance(structured, Mapping) and structured.get("event"):
                for key, value in structured.items():
                    if key in {"message", "created_at", "level", "logger"}:
                        continue
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        snapshot[str(key)] = value
                snapshot["structured"] = True
            else:
                snapshot["structured"] = False
            with self.lock:
                self._records.append(snapshot)
        except Exception:
            self.handleError(record)

    def recent(
        self,
        *,
        limit: int = 100,
        level: str | None = None,
        logger_name: str | None = None,
        event: str | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        """按时间倒序返回脱敏日志快照。"""

        safe_limit = max(0, min(int(limit), 1000))
        if safe_limit == 0:
            return ()
        level_key = str(level or "").strip().upper()
        logger_key = str(logger_name or "").strip()
        event_key = str(event or "").strip()
        result = []
        with self.lock:
            for item in reversed(self._records):
                if level_key and str(item.get("level", "")).upper() != level_key:
                    continue
                if logger_key and logger_key not in str(item.get("logger", "")):
                    continue
                if event_key and str(item.get("event", "")) != event_key:
                    continue
                result.append(dict(item))
                if len(result) >= safe_limit:
                    break
        return tuple(result)

    def clear(self) -> None:
        with self.lock:
            self._records.clear()

    @property
    def capacity(self) -> int:
        return self._records.maxlen or 0


def _is_tty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _mark_handler(handler: logging.Handler, kind: str) -> logging.Handler:
    setattr(handler, _HANDLER_MARKER, kind)
    return handler


def _close_owned_handlers(root: logging.Logger, *, force: bool) -> None:
    global _MEMORY_HANDLER
    for handler in tuple(root.handlers):
        owned = getattr(handler, _HANDLER_MARKER, None)
        if owned is None and not force:
            continue
        root.removeHandler(handler)
        try:
            handler.close()
        except (OSError, ValueError):
            continue
        if handler is _MEMORY_HANDLER:
            _MEMORY_HANDLER = None


def recent_log_records(
    *,
    limit: int = 100,
    level: str | None = None,
    logger_name: str | None = None,
    event: str | None = None,
) -> tuple[Mapping[str, object], ...]:
    """读取本地日志环中最近记录；没有内存 sink 时返回空元组。"""

    handler = _MEMORY_HANDLER
    if handler is None:
        return ()
    return handler.recent(
        limit=limit,
        level=level,
        logger_name=logger_name,
        event=event,
    )


def clear_log_records() -> None:
    """清空本地内存日志环，不触碰文件日志。"""

    handler = _MEMORY_HANDLER
    if handler is not None:
        handler.clear()


class _FileErrorReportingMixin:
    """文件写入失败时只向 lastResort 输出一次脱敏结构化终态。"""

    _meapet_write_failure_reported = False

    def handleError(self, record: logging.LogRecord) -> None:
        if self._meapet_write_failure_reported:
            return
        self._meapet_write_failure_reported = True
        fallback = logging.lastResort
        if fallback is None:
            return
        error_type = getattr(sys.exc_info()[0], "__name__", "Exception")
        payload = json.dumps(
            {
                "component": "logger",
                "error_type": sanitize_log_text(error_type, limit=96),
                "event": "logging.sink.write_failed",
                "reason_code": "file_write_failed",
                "status": "degraded",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        failure_record = logging.LogRecord(
            "logger",
            logging.ERROR,
            "",
            0,
            payload,
            (),
            None,
        )
        try:
            fallback.handle(failure_record)
        except (OSError, RuntimeError, ValueError):
            return


class _SafeRotatingFileHandler(
    _FileErrorReportingMixin,
    logging.handlers.RotatingFileHandler,
):
    pass


class _SafeTimedRotatingFileHandler(
    _FileErrorReportingMixin,
    logging.handlers.TimedRotatingFileHandler,
):
    pass


class _SafeFileHandler(
    _FileErrorReportingMixin,
    logging.FileHandler,
):
    pass


class _LogFileSetupError(RuntimeError):
    """日志文件处理器无法创建；异常正文不跨越初始化边界。"""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = str(reason_code or "file_handler_unavailable")


def _file_handler(settings: LoggingSettings) -> logging.Handler | None:
    if not settings.file_enabled:
        return None
    path = settings.file_path or Path("./data/meapet.log").resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if settings.rotation == "size":
            handler: logging.Handler = _SafeRotatingFileHandler(
                path,
                maxBytes=settings.max_bytes,
                backupCount=settings.backup_count,
                encoding="utf-8",
            )
        elif settings.rotation == "time":
            handler = _SafeTimedRotatingFileHandler(
                path,
                when=settings.when,
                interval=settings.interval,
                backupCount=settings.backup_count,
                encoding="utf-8",
                utc=settings.utc,
            )
        else:
            handler = _SafeFileHandler(path, encoding="utf-8")
    except OSError as exc:
        raise _LogFileSetupError("file_open_failed") from exc
    except ValueError as exc:
        raise _LogFileSetupError("file_rotation_invalid") from exc
    handler.setLevel(settings.file_level)
    handler.setFormatter(
        SafeFormatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return _mark_handler(handler, "file")


def configure_logging(
    level: str | int | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    base_directory: str | Path | None = None,
    log_path: str | Path | None = None,
    console: bool | None = None,
    color: bool | str | None = None,
    rotation: str | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
    when: str | None = None,
    interval: int | None = None,
    utc: bool | None = None,
    stream: TextIO | None = None,
    force: bool = False,
    emit_event: bool = True,
) -> logging.Logger:
    """配置根 logger 并返回它。

    ``config`` 可以是完整的配置映射，也可以直接是 ``logging`` 子映射。文件
    默认关闭；提供 ``log_path`` 或 ``logging.file.path`` 后自动开启。轮换方式
    支持 ``size``、``time`` 和 ``none``，文件保留数量由 ``backup_count`` 控制。
    文件创建失败时保留控制台日志，不阻断桌宠启动。
    """

    base = Path(base_directory).expanduser().resolve() if base_directory is not None else None
    settings = LoggingSettings.from_mapping(
        config,
        base_directory=base,
        level=level,
        log_path=log_path,
        console=console,
        color=color,
        rotation=rotation,
        max_bytes=max_bytes,
        backup_count=backup_count,
        when=when,
        interval=interval,
        utc=utc,
    )
    output = stream if stream is not None else sys.stderr
    # ANSI 只允许写入真正的终端；即使配置显式开启，重定向到文件/管道也不污染输出。
    color_enabled = _is_tty(output) and (settings.color if settings.color is not None else True)
    enabled_levels = [
        level_value
        for level_value, enabled in (
            (settings.console_level, settings.console_enabled),
            (settings.file_level, settings.file_enabled),
            (settings.memory_level, settings.memory_enabled),
        )
        if enabled
    ]
    minimum_level = min(enabled_levels) if enabled_levels else settings.level

    root = logging.getLogger()
    file_failure_code = ""
    memory_handler: MemoryLogHandler | None = None
    with _CONFIG_LOCK:
        _close_owned_handlers(root, force=force)
        root.setLevel(minimum_level)
        # asyncio 在 debug 模式会把子进程 transport 的内部生命周期以 INFO
        # 级别写入根 logger；这些行既不是桌宠业务事件，也不是统一 JSON
        # 事件，默认提升到 WARNING，避免污染控制台和结构化日志 sink。
        asyncio_logger = logging.getLogger("asyncio")
        if asyncio_logger.level == logging.NOTSET:
            asyncio_logger.setLevel(logging.WARNING)
        if settings.console_enabled:
            console_handler = logging.StreamHandler(output)
            console_handler.setLevel(settings.console_level)
            console_handler.setFormatter(ColorFormatter(enabled=bool(color_enabled)))
            root.addHandler(_mark_handler(console_handler, "console"))
        try:
            file_handler = _file_handler(settings)
        except _LogFileSetupError as exc:
            file_handler = None
            file_failure_code = exc.reason_code
        if file_handler is not None:
            root.addHandler(file_handler)
        if settings.memory_enabled:
            memory_handler = MemoryLogHandler(settings.memory_capacity)
            memory_handler.setLevel(settings.memory_level)
            root.addHandler(_mark_handler(memory_handler, "memory"))
            global _MEMORY_HANDLER
            _MEMORY_HANDLER = memory_handler
    setup_logger = logging.getLogger(__name__)
    if file_failure_code:
        log_event(
            setup_logger,
            "logging.sink.failed",
            component="logger",
            status="degraded",
            level=logging.WARNING,
            reason_code=file_failure_code,
            detail="轮换日志文件不可用，应用继续使用其它日志输出",
            fields={"sink": "file"},
        )
    if emit_event:
        log_event(
            setup_logger,
            "logging.configure.completed",
            component="logger",
            status="completed",
            fields={
                "console_enabled": settings.console_enabled,
                "console_level": settings.console_level,
                "color_enabled": bool(color_enabled),
                "file_enabled": settings.file_enabled,
                "file_active": file_handler is not None,
                "file_level": settings.file_level,
                "rotation": settings.rotation,
                "memory_enabled": settings.memory_enabled,
                "memory_capacity": settings.memory_capacity,
            },
        )
    return root


def close_logging() -> None:
    """关闭并移除本模块创建的处理器，保留宿主安装的处理器。"""

    with _CONFIG_LOCK:
        _close_owned_handlers(logging.getLogger(), force=False)
