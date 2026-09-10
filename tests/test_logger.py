from __future__ import annotations

import io
import json
import logging
import logging.handlers
from pathlib import Path

import pytest

from logger import (
    ColorFormatter,
    SafeFormatter,
    clear_log_records,
    close_logging,
    configure_logging,
    event_fingerprint,
    log_event,
    recent_log_records,
)


class _TerminalBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def clean_logging():
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    for handler in previous_handlers:
        root.removeHandler(handler)
    try:
        yield root
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)


def test_console_levels_and_terminal_colors(clean_logging) -> None:
    stream = _TerminalBuffer()
    logger = configure_logging(level="DEBUG", stream=stream, color=True)

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")

    output = stream.getvalue()
    assert "[DEBUG]" in output
    assert "info message" in output
    assert "\x1b[" in output


def test_colors_are_removed_for_non_terminal_stream(clean_logging) -> None:
    stream = io.StringIO()
    logger = configure_logging(level="INFO", stream=stream, color=True)

    logger.info("plain message")

    assert "plain message" in stream.getvalue()
    assert "\x1b[" not in stream.getvalue()


def test_memory_log_sink_keeps_bounded_redacted_structured_records(clean_logging) -> None:
    configure_logging(
        config={
            "console": {"enabled": False},
            "memory": {"enabled": True, "capacity": 2},
        },
        emit_event=False,
    )
    clear_log_records()
    target = logging.getLogger("memory-test")
    target.info("first")
    log_event(
        target,
        "memory.test.completed",
        component="memory-test",
        status="completed",
        fields={"count": 2, "api_key": "secret-value"},
    )
    target.warning("third token: secret-value")

    records = recent_log_records(limit=10, logger_name="memory-test")
    assert len(records) == 2
    assert records[0]["message"] == "third [redacted]"
    assert records[1]["event"] == "memory.test.completed"
    assert records[1]["count"] == 2
    assert "secret-value" not in repr(records)
    assert recent_log_records(limit=0) == ()
    assert recent_log_records(limit=-1) == ()
    assert recent_log_records(limit=1) == records[:1]
    assert recent_log_records(limit=1, level="INFO", event="memory.test.completed") == records[1:]
    records[0]["message"] = "edited snapshot"
    assert recent_log_records(limit=1)[0]["message"] == "third [redacted]"


def test_console_color_accepts_ui_always_and_never_values(clean_logging) -> None:
    always = configure_logging(config={"console": {"color": "always"}}, stream=_TerminalBuffer())
    console_handler = next(
        handler
        for handler in always.handlers
        if getattr(handler, "_meapet_logging_handler", "") == "console"
    )
    formatter = console_handler.formatter
    assert isinstance(formatter, ColorFormatter)
    assert formatter.enabled is True

    never_stream = _TerminalBuffer()
    never_logger = configure_logging(config={"console": {"color": "never"}}, stream=never_stream)
    never_logger.info("plain message")
    assert "\x1b[" not in never_stream.getvalue()

    false_stream = _TerminalBuffer()
    false_logger = configure_logging(config={"console": {"color": False}}, stream=false_stream)
    false_logger.info("plain message")
    assert "\x1b[" not in false_stream.getvalue()

    with pytest.raises(ValueError, match="logging.console.color"):
        configure_logging(config={"console": {"color": "sometimes"}})


def test_console_and_file_levels_are_independent(clean_logging, tmp_path: Path) -> None:
    stream = _TerminalBuffer()
    path = tmp_path / "levels.log"
    logger = configure_logging(
        config={
            "level": "DEBUG",
            "console": {"enabled": True, "level": "WARNING", "color": True},
            "file": {
                "enabled": True,
                "path": str(path),
                "level": "DEBUG",
                "rotation": "none",
            },
        },
        stream=stream,
    )

    logger.debug("debug only in file")
    logger.warning("warning in both sinks")
    for handler in logger.handlers:
        handler.flush()

    console_output = stream.getvalue()
    file_output = path.read_text(encoding="utf-8")
    assert "debug only in file" not in console_output
    assert "warning in both sinks" in console_output
    assert "debug only in file" in file_output
    assert "warning in both sinks" in file_output
    assert "\x1b[" not in file_output


def test_invalid_level_is_rejected(clean_logging) -> None:
    with pytest.raises(ValueError, match="logging level"):
        configure_logging(config={"level": "TRACE"})


@pytest.mark.parametrize(
    ("section", "value", "message"),
    (("console", "invalid", "logging.console"), ("file", 42, "logging.file")),
)
def test_invalid_logging_section_type_is_rejected(
    clean_logging, section: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        configure_logging(config={section: value})


def test_invalid_file_path_and_rotation_settings_are_rejected(clean_logging) -> None:
    with pytest.raises(ValueError, match="logging.file.path"):
        configure_logging(config={"file": {"enabled": True, "path": 42}})
    with pytest.raises(ValueError, match="logging.file.when"):
        configure_logging(config={"file": {"enabled": True, "path": "pet.log", "when": "unknown"}})


def test_size_rotation_and_retention(clean_logging, tmp_path: Path) -> None:
    path = tmp_path / "logs" / "pet.log"
    logger = configure_logging(
        config={
            "level": "INFO",
            "console": {"enabled": False},
            "file": {
                "enabled": True,
                "path": str(path),
                "rotation": "size",
                "max_bytes": 180,
                "backup_count": 2,
            },
        }
    )

    for index in range(30):
        logger.info("message-%s %s", index, "x" * 80)
    for handler in logger.handlers:
        handler.flush()

    files = sorted(path.parent.glob("pet.log*"))
    assert path.is_file()
    assert (path.parent / "pet.log.1").is_file()
    assert len(files) <= 3


def test_time_rotation_and_relative_path(clean_logging, tmp_path: Path) -> None:
    logger = configure_logging(
        config={
            "logging": {
                "console": {"enabled": False},
                "file": {
                    "enabled": True,
                    "path": "logs/pet.log",
                    "rotation": "time",
                    "when": "S",
                    "interval": 2,
                    "backup_count": 4,
                },
            }
        },
        base_directory=tmp_path,
    )

    handlers = [handler for handler in logger.handlers if isinstance(handler, logging.Handler)]
    file_handler = next(handler for handler in handlers if hasattr(handler, "baseFilename"))
    assert isinstance(file_handler, logging.handlers.TimedRotatingFileHandler)
    assert Path(file_handler.baseFilename) == (tmp_path / "logs" / "pet.log").resolve()
    assert file_handler.backupCount == 4


def test_reconfiguration_closes_old_file_and_switches_path(clean_logging, tmp_path: Path) -> None:
    first_path = tmp_path / "first.log"
    second_path = tmp_path / "second.log"
    logger = configure_logging(
        config={
            "console": {"enabled": False},
            "file": {"path": str(first_path), "rotation": "none"},
        }
    )
    first_handler = next(handler for handler in logger.handlers if hasattr(handler, "baseFilename"))
    first_stream = first_handler.stream
    assert first_stream is not None

    configure_logging(
        config={
            "console": {"enabled": False},
            "file": {"path": str(second_path), "rotation": "none"},
        }
    )
    logger.info("switched")
    for handler in logger.handlers:
        handler.flush()

    assert first_stream.closed
    assert second_path.read_text(encoding="utf-8").endswith("switched\n")
    owned_file_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_meapet_logging_handler", None) == "file"
    ]
    assert len(owned_file_handlers) == 1


def test_close_logging_releases_owned_file_handler(clean_logging, tmp_path: Path) -> None:
    path = tmp_path / "close.log"
    logger = configure_logging(
        config={"console": {"enabled": False}, "file": {"path": str(path), "rotation": "none"}}
    )
    handler = next(handler for handler in logger.handlers if hasattr(handler, "baseFilename"))
    stream = handler.stream
    assert stream is not None

    close_logging()

    assert stream.closed
    assert not any(getattr(item, "_meapet_logging_handler", None) for item in logger.handlers)


def test_structured_event_fingerprints_ids_and_redacts_private_fields() -> None:
    stream = io.StringIO()
    isolated = logging.Logger("structured-test", level=logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    isolated.addHandler(handler)
    secret = "sk-private-logger-token"
    operation_id = "operation-private-id"
    request_id = "request-private-id"

    log_event(
        isolated,
        "test.operation.completed",
        component="test",
        status="completed",
        correlation_id="correlation-private-id",
        operation_id=operation_id,
        request_id=request_id,
        reason_code="worker_ready",
        detail=(
            f"token={secret} endpoint=https://private.invalid/v1 path=/home/private/work/model.bin"
        ),
        fields={
            "api_key": secret,
            "prompt": "private prompt body",
            "path": "/home/private/work/model.bin",
            "stderr": "private worker output",
            "stderr_fingerprint": "abc123",
            "stderr_bytes": 37,
            "request_id": "nested-private-request",
        },
    )

    payload = json.loads(stream.getvalue())
    assert payload["correlation"] == event_fingerprint("correlation-private-id")
    assert payload["operation"] == event_fingerprint(operation_id)
    assert payload["request"] == event_fingerprint(request_id)
    assert payload["reason_code"] == "worker_ready"
    assert payload["api_key"] == "[redacted]"
    assert payload["prompt"] == "[redacted]"
    assert payload["path"] == "[redacted]"
    assert payload["stderr"] == "[redacted]"
    assert payload["stderr_fingerprint"] == "abc123"
    assert payload["stderr_bytes"] == 37
    assert payload["request_id"] == event_fingerprint("nested-private-request")
    rendered = stream.getvalue()
    assert secret not in rendered
    assert "private.invalid" not in rendered
    assert "/home/private" not in rendered
    assert "[url]" in payload["detail"]
    assert "[path]" in payload["detail"]


def test_safe_formatter_keeps_exception_type_without_secret_or_absolute_path() -> None:
    stream = io.StringIO()
    isolated = logging.Logger("safe-exception-test", level=logging.DEBUG)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SafeFormatter("%(levelname)s %(message)s"))
    isolated.addHandler(handler)
    secret = "sk-private-exception-token"

    try:
        raise RuntimeError(
            f"token={secret} failed at /home/private/work/file.py https://private.invalid/failure"
        )
    except RuntimeError:
        isolated.exception(
            "worker failed token=%s path=/home/private/work/file.py",
            secret,
        )

    rendered = stream.getvalue()
    assert "ERROR worker failed" in rendered
    assert "exception_type=RuntimeError" in rendered
    assert "reason_fingerprint=" in rendered
    assert "test_logger.py:" in rendered
    assert secret not in rendered
    assert "private.invalid" not in rendered
    assert "/home/private" not in rendered


def test_file_sink_failure_is_visible_and_does_not_log_private_path(
    clean_logging,
    monkeypatch,
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    private_path = tmp_path / "private-user" / "meapet.log"

    def reject_file(*_args, **_kwargs):
        raise PermissionError("private path denied")

    monkeypatch.setattr("logger._SafeFileHandler", reject_file)
    logger = configure_logging(
        config={
            "console": {"enabled": True, "level": "INFO", "color": False},
            "file": {
                "enabled": True,
                "path": str(private_path),
                "rotation": "none",
            },
        },
        stream=stream,
    )
    for handler in logger.handlers:
        handler.flush()

    rendered = stream.getvalue()
    assert '"event": "logging.sink.failed"' in rendered
    assert '"reason_code": "file_open_failed"' in rendered
    assert '"file_active": false' in rendered
    assert str(private_path) not in rendered


def test_file_write_failure_uses_one_bounded_last_resort_event(
    clean_logging,
    monkeypatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "runtime-write.log"
    logger = configure_logging(
        config={
            "console": {"enabled": False},
            "file": {
                "enabled": True,
                "path": str(log_path),
                "rotation": "none",
            },
        }
    )
    file_handler = next(
        handler
        for handler in logger.handlers
        if getattr(handler, "_meapet_logging_handler", "") == "file"
    )
    fallback_stream = io.StringIO()
    fallback = logging.StreamHandler(fallback_stream)
    fallback.setFormatter(logging.Formatter("%(message)s"))
    monkeypatch.setattr(logging, "lastResort", fallback)
    record = logging.LogRecord(
        "private.source",
        logging.ERROR,
        "/home/private/source.py",
        42,
        "private body",
        (),
        None,
    )

    for _index in range(2):
        try:
            raise OSError("token=sk-private-file-write /home/private/runtime.log")
        except OSError:
            file_handler.handleError(record)

    rendered = fallback_stream.getvalue()
    assert rendered.count("logging.sink.write_failed") == 1
    assert '"reason_code": "file_write_failed"' in rendered
    assert '"error_type": "OSError"' in rendered
    assert "sk-private-file-write" not in rendered
    assert "/home/private" not in rendered
