from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any

from core.events.types import AudioChunk
from logger.events import log_event

logger = logging.getLogger(__name__)


def _device_id_text(device: object) -> str:
    """把 Qt 输出设备 ID 转成可精确持久化的文本。"""

    identifier = getattr(device, "id", None)
    if not callable(identifier):
        return ""
    try:
        raw = bytes(identifier())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"hex:{raw.hex()}"


def _device_description(device: object) -> str:
    """返回本地设备选择器可显示的有界描述。"""

    description = getattr(device, "description", None)
    if not callable(description):
        return "未命名扬声器"
    try:
        rendered = " ".join(str(description() or "").replace("\x00", " ").split())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return "未命名扬声器"
    return rendered[:160] or "未命名扬声器"


def _normalize_device_id(value: object) -> str:
    """校验配置中的精确输出设备 ID。"""

    device_id = str(value or "")
    if len(device_id) > 1_024 or any(char in device_id for char in "\x00\r\n"):
        raise ValueError("audio output device id is invalid")
    return device_id


@dataclass(frozen=True, slots=True)
class AudioOutputDeviceChoice:
    """仅供本地配置控件使用的输出设备选项。"""

    device_id: str = field(repr=False)
    description: str
    is_default: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", _normalize_device_id(self.device_id))
        if not self.device_id:
            raise ValueError("audio output device id is required")
        description = " ".join(str(self.description or "").replace("\x00", " ").split())
        object.__setattr__(self, "description", description[:160] or "未命名扬声器")


@dataclass
class _PlaybackStream:
    """同一请求的连续 PCM；不同上下文绝不交错写入一个设备。"""

    context: object
    request_id: str
    sample_rate: int
    channels: int
    sample_format: str
    buffers: deque[bytes] = field(default_factory=deque)
    head_offset: int = 0
    pending_bytes: int = 0
    final: bool = False

    @property
    def key(self) -> tuple[object, str, int, int, str]:
        return (
            self.context,
            self.request_id,
            self.sample_rate,
            self.channels,
            self.sample_format,
        )

    @classmethod
    def from_chunk(cls, chunk: AudioChunk) -> _PlaybackStream:
        stream = cls(
            chunk.context,
            chunk.request_id,
            chunk.sample_rate,
            chunk.channels,
            chunk.sample_format,
        )
        stream.append(chunk)
        return stream

    def append(self, chunk: AudioChunk) -> None:
        if chunk.data:
            data = bytes(chunk.data)
            self.buffers.append(data)
            self.pending_bytes += len(data)
        self.final = self.final or bool(chunk.is_final)

    def peek(self, maximum: int) -> bytes:
        if not self.buffers or maximum <= 0:
            return b""
        head = self.buffers[0]
        return head[self.head_offset : self.head_offset + maximum]

    def consume(self, count: int) -> None:
        remaining = max(0, min(int(count), self.pending_bytes))
        self.pending_bytes -= remaining
        while remaining and self.buffers:
            available = len(self.buffers[0]) - self.head_offset
            if remaining < available:
                self.head_offset += remaining
                return
            remaining -= available
            self.buffers.popleft()
            self.head_offset = 0


@dataclass(frozen=True)
class PlaybackReceipt:
    """一个物理播放流的终态回执；正文和上下文只在进程内回传。"""

    context: object = field(repr=False)
    request_id: str
    state: str
    bytes_written: int = 0
    duration_ms: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    reason_code: str = ""

    def as_dict(self) -> dict[str, object]:
        """返回可交给 UI/诊断层的回执，不暴露上下文对象。"""

        return {
            "request_id": self.request_id,
            "state": self.state,
            "bytes_written": max(0, int(self.bytes_written)),
            "duration_ms": max(0.0, round(float(self.duration_ms), 3)),
            "sample_rate": max(0, int(self.sample_rate)),
            "channels": max(0, int(self.channels)),
            "reason_code": self.reason_code,
        }


try:
    from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal
    from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices

    _QT_AUDIO_AVAILABLE = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):
    _QT_AUDIO_AVAILABLE = False


if _QT_AUDIO_AVAILABLE:

    class QtAudioPlayer(QObject):
        """把同一请求的 S16LE 分片持续写入一个 QAudioSink。"""

        audioReceived = Signal(object)
        cancelRequested = Signal(object)
        stopAllRequested = Signal()
        playbackStateChanged = Signal(bool)
        playbackReceipt = Signal(object)

        def __init__(
            self,
            parent: QObject | None = None,
            *,
            max_pending_bytes: int = 32 * 1024 * 1024,
            max_pending_streams: int = 256,
            max_stalled_drains: int = 200,
            max_completion_polls: int = 6000,
            output_device_provider: Callable[[], object | None] | None = None,
            sink_factory: Callable[[object, object, QObject], object] | None = None,
            drain_scheduler: Callable[[Callable[[], None]], object] | None = None,
            output_device_id: str = "",
            media_devices: object | None = None,
        ) -> None:
            super().__init__(parent)
            self.audioReceived.connect(self._enqueue)
            self.cancelRequested.connect(self._cancel_context)
            self.stopAllRequested.connect(self._stop_all)
            if (
                isinstance(max_pending_bytes, bool)
                or not 4096 <= int(max_pending_bytes) <= 256 * 1024 * 1024
            ):
                raise ValueError("audio playback queue limit is invalid")
            if isinstance(max_pending_streams, bool) or not 1 <= int(max_pending_streams) <= 4096:
                raise ValueError("audio playback stream limit is invalid")
            if isinstance(max_stalled_drains, bool) or not 1 <= int(max_stalled_drains) <= 10000:
                raise ValueError("audio playback stall limit is invalid")
            if (
                isinstance(max_completion_polls, bool)
                or not 10 <= int(max_completion_polls) <= 12000
            ):
                raise ValueError("audio playback completion limit is invalid")
            self._max_pending_bytes = int(max_pending_bytes)
            self._max_pending_streams = int(max_pending_streams)
            self._max_stalled_drains = int(max_stalled_drains)
            self._max_completion_polls = int(max_completion_polls)
            self._uses_custom_output_provider = output_device_provider is not None
            self._output_device_id = _normalize_device_id(output_device_id)
            self._media_devices = media_devices
            if self._media_devices is None and QCoreApplication.instance() is not None:
                try:
                    self._media_devices = QMediaDevices(self)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    self._media_devices = None
            self._output_device_provider = output_device_provider or self._media_output_device
            self._sink_factory = sink_factory or (
                lambda device, audio_format, owner: QAudioSink(device, audio_format, owner)
            )
            self._drain_scheduler = drain_scheduler or (
                lambda callback: QTimer.singleShot(10, callback)
            )
            self._pending: deque[_PlaybackStream] = deque()
            self._current: _PlaybackStream | None = None
            self._sink: Any | None = None
            self._write_device: Any | None = None
            self._current_context: object | None = None
            self._cancelled_contexts: set[object] = set()
            self._cancelled_lock = Lock()
            self._epoch_lock = Lock()
            self._playback_epoch = 0
            self._queued_bytes = 0
            self._dropped_streams = 0
            self._last_error = ""
            self._drain_scheduled = False
            self._completion_scheduled = False
            self._sink_generation = 0
            self._write_generation = 0
            self._active_write_generation = 0
            self._idle_write_generation = 0
            self._written_bytes = 0
            self._stalled_drains = 0
            self._completion_polls = 0
            self._activity = False
            self._closed = False
            # QAudioSink/QIODevice 允许在 write() 内同步发出 stateChanged。
            # 该标志禁止同一 PCM 在写入尚未消费前被重入 drain。
            self._drain_in_progress = False
            self._drain_requested = False
            self._active_state_seen = False
            self._stream_started_at = 0.0
            self._active_device_id = ""
            self._device_listener_connected = False
            self._connect_device_listener()

        @property
        def available(self) -> bool:
            return not self._closed and self._resolve_output_device() is not None

        @property
        def playing(self) -> bool:
            return bool(self._current is not None or self._pending or self._sink is not None)

        @staticmethod
        def _default_output_device() -> object | None:
            if QCoreApplication.instance() is None:
                return None
            try:
                device = QMediaDevices.defaultAudioOutput()
                return None if device.isNull() else device
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return None

        def _media_output_device(self) -> object | None:
            """按精确 ID 解析输出设备；空 ID 使用系统默认设备。"""

            media_devices = self._media_devices
            if media_devices is None:
                return self._default_output_device()
            if self._output_device_id:
                getter = getattr(media_devices, "audioOutputs", None)
                try:
                    devices = tuple(getter()) if callable(getter) else ()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    devices = ()
                for device in devices:
                    if _device_id_text(device) == self._output_device_id:
                        is_null = getattr(device, "isNull", None)
                        if callable(is_null) and bool(is_null()):
                            return None
                        return device
                return None
            getter = getattr(media_devices, "defaultAudioOutput", None)
            try:
                device = getter() if callable(getter) else None
                is_null = getattr(device, "isNull", None)
                if device is not None and callable(is_null) and bool(is_null()):
                    return None
                return device
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return None

        def _connect_device_listener(self) -> None:
            if self._uses_custom_output_provider or self._device_listener_connected:
                return
            changed = getattr(self._media_devices, "audioOutputsChanged", None)
            connect = getattr(changed, "connect", None)
            if callable(connect):
                try:
                    connect(self._audio_outputs_changed)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    return
                self._device_listener_connected = True

        def _disconnect_device_listener(self) -> None:
            if not self._device_listener_connected:
                return
            changed = getattr(self._media_devices, "audioOutputsChanged", None)
            disconnect = getattr(changed, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect(self._audio_outputs_changed)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            self._device_listener_connected = False

        def _audio_outputs_changed(self, *_args: object) -> None:
            """设备热插拔后立即收敛活动 sink，并唤醒排队流。"""

            if self._closed:
                return
            getter = getattr(self._media_devices, "audioOutputs", None)
            try:
                devices = tuple(getter()) if callable(getter) else ()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                devices = ()
            available_ids = {_device_id_text(device) for device in devices}
            current = self._current
            if current is None:
                if self._pending:
                    self._play_next()
                return
            # 默认设备发生切换但旧设备仍存在时，不打断当前语音；只在
            # 实际承载当前流的设备消失，或显式选中的设备消失时失败。
            active_missing = bool(
                self._active_device_id and self._active_device_id not in available_ids
            )
            selected_missing = bool(
                self._output_device_id and self._output_device_id not in available_ids
            )
            if active_missing or selected_missing:
                self._fail_current("audio_output_disconnected")

        @property
        def output_device_id(self) -> str:
            """返回当前精确输出设备 ID；空值表示系统默认设备。"""

            return self._output_device_id

        def set_output_device_id(self, device_id: object) -> str:
            """切换精确输出设备；活动流在设备改变时明确失败并收敛。"""

            normalized = _normalize_device_id(device_id)
            if normalized == self._output_device_id:
                return normalized
            self._output_device_id = normalized
            if self._current is not None:
                resolved = self._resolve_output_device()
                resolved_id = _device_id_text(resolved) if resolved is not None else ""
                if resolved is None or (
                    self._active_device_id and resolved_id and resolved_id != self._active_device_id
                ):
                    self._fail_current("audio_output_changed")
            elif self._pending:
                self._play_next()
            return normalized

        def output_devices(self) -> tuple[AudioOutputDeviceChoice, ...]:
            """枚举当前输出设备，供配置界面精确选择。"""

            if self._closed or self._media_devices is None:
                return ()
            getter = getattr(self._media_devices, "audioOutputs", None)
            default_getter = getattr(self._media_devices, "defaultAudioOutput", None)
            try:
                devices = tuple(getter()) if callable(getter) else ()
                default = default_getter() if callable(default_getter) else None
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return ()
            default_id = _device_id_text(default) if default is not None else ""
            choices: list[AudioOutputDeviceChoice] = []
            seen: set[str] = set()
            for device in devices:
                device_id = _device_id_text(device)
                if not device_id or device_id in seen:
                    continue
                seen.add(device_id)
                choices.append(
                    AudioOutputDeviceChoice(
                        device_id=device_id,
                        description=_device_description(device),
                        is_default=bool(default_id and device_id == default_id),
                    )
                )
            return tuple(choices)

        def _resolve_output_device(self) -> object | None:
            try:
                device = self._output_device_provider()
                if device is None:
                    return None
                is_null = getattr(device, "isNull", None)
                if callable(is_null) and bool(is_null()):
                    return None
                return device
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return None

        def diagnostics(self) -> dict[str, object]:
            """返回不含设备名称、ID、音频正文或上下文的公开状态。"""

            available = self.available
            return {
                "status": "active" if self.playing else "ready" if available else "unavailable",
                "available": available,
                "active": self.playing,
                "queued_streams": len(self._pending) + (1 if self._current is not None else 0),
                "queued_bytes": max(0, int(self._queued_bytes)),
                "dropped_streams": max(0, int(self._dropped_streams)),
                "error_code": self._last_error,
            }

        def __call__(self, chunk: AudioChunk) -> None:
            if not self._closed and isinstance(chunk, AudioChunk):
                with self._epoch_lock:
                    epoch = self._playback_epoch
                self.audioReceived.emit((epoch, chunk))

        def _enqueue(self, payload: object) -> None:
            if not isinstance(payload, tuple) or len(payload) != 2:
                return
            epoch, chunk = payload
            if not isinstance(epoch, int) or not isinstance(chunk, AudioChunk):
                return
            with self._epoch_lock:
                if epoch != self._playback_epoch:
                    return
            if self._closed or self._is_cancelled(chunk.context):
                return
            if not chunk.data and not chunk.is_final:
                return
            incoming_bytes = len(chunk.data)
            if self._queued_bytes + incoming_bytes > self._max_pending_bytes:
                affected = sum(stream.context == chunk.context for stream in self._pending)
                affected += int(
                    self._current is not None and self._current.context == chunk.context
                )
                self._dropped_streams += max(1, affected)
                self._last_error = "queue_limit"
                with self._cancelled_lock:
                    self._cancelled_contexts.add(chunk.context)
                self._cancel_context(chunk.context)
                log_event(
                    logger,
                    "audio.playback.queue_overflow",
                    component="gui.qt6.audio",
                    status="failed",
                    level=logging.WARNING,
                    correlation_id=chunk.context,
                    reason_code="queue_limit",
                    fields={"queue_limit": self._max_pending_bytes},
                )
                return

            key = (
                chunk.context,
                chunk.request_id,
                chunk.sample_rate,
                chunk.channels,
                chunk.sample_format,
            )
            if self._current is not None and self._current.key == key:
                target = self._current
            else:
                target = next((stream for stream in self._pending if stream.key == key), None)
            if target is None:
                if len(self._pending) + int(self._current is not None) >= self._max_pending_streams:
                    affected = sum(stream.context == chunk.context for stream in self._pending)
                    affected += int(
                        self._current is not None and self._current.context == chunk.context
                    )
                    self._dropped_streams += max(1, affected)
                    self._last_error = "stream_limit"
                    with self._cancelled_lock:
                        self._cancelled_contexts.add(chunk.context)
                    self._cancel_context(chunk.context)
                    log_event(
                        logger,
                        "audio.playback.queue_overflow",
                        component="gui.qt6.audio",
                        status="failed",
                        level=logging.WARNING,
                        correlation_id=chunk.context,
                        reason_code="stream_limit",
                        fields={"stream_limit": self._max_pending_streams},
                    )
                    return
                target = _PlaybackStream.from_chunk(chunk)
                self._pending.append(target)
                self._queued_bytes += incoming_bytes
                self._sync_activity()
                if self._current is None:
                    self._play_next()
                return
            target.append(chunk)
            self._queued_bytes += incoming_bytes
            self._sync_activity()
            if target is self._current:
                self._drain_current()
                self._maybe_finish_current()

        @staticmethod
        def _format_for(stream: _PlaybackStream) -> object:
            audio_format = QAudioFormat()
            audio_format.setSampleRate(stream.sample_rate)
            audio_format.setChannelCount(stream.channels)
            sample_format = getattr(getattr(QAudioFormat, "SampleFormat", None), "Int16", None)
            if sample_format is None:
                sample_format = getattr(QAudioFormat, "Int16", None)
            if sample_format is None:
                raise RuntimeError("audio_format_unavailable")
            audio_format.setSampleFormat(sample_format)
            return audio_format

        def _play_next(self) -> None:
            if self._closed or self._current is not None:
                return
            while self._pending:
                stream = self._pending.popleft()
                if self._is_cancelled(stream.context):
                    self._queued_bytes = max(0, self._queued_bytes - stream.pending_bytes)
                    continue
                if stream.pending_bytes == 0 and stream.final:
                    continue
                self._current = stream
                self._current_context = stream.context
                self._write_generation = 0
                self._active_write_generation = 0
                self._idle_write_generation = 0
                self._written_bytes = 0
                self._stalled_drains = 0
                self._completion_polls = 0
                self._drain_in_progress = False
                self._drain_requested = False
                self._active_state_seen = False
                self._stream_started_at = 0.0
                self._active_device_id = ""
                self._sink_generation += 1
                sink_generation = self._sink_generation
                sink: object | None = None
                try:
                    device = self._resolve_output_device()
                    if device is None:
                        raise RuntimeError("audio_output_unavailable")
                    self._active_device_id = _device_id_text(device)
                    audio_format = self._format_for(stream)
                    supports = getattr(device, "isFormatSupported", None)
                    if callable(supports) and not bool(supports(audio_format)):
                        raise RuntimeError("audio_format_unsupported")
                    sink = self._sink_factory(device, audio_format, self)
                    self._sink = sink
                    state_changed = getattr(sink, "stateChanged", None)
                    connect = getattr(state_changed, "connect", None)
                    if callable(connect):
                        connect(
                            lambda state, owner=sink, generation=sink_generation: (
                                self._state_changed(owner, generation, state)
                            )
                        )
                    write_device = sink.start()
                    if write_device is None or not callable(getattr(write_device, "write", None)):
                        raise RuntimeError("audio_output_start_failed")
                    if (
                        sink_generation != self._sink_generation
                        or sink is not self._sink
                        or stream is not self._current
                    ):
                        return
                    self._write_device = write_device
                    self._stream_started_at = monotonic()
                    self._last_error = ""
                    self._sync_activity()
                    self._emit_receipt(stream, "started")
                    self._drain_current()
                    self._maybe_finish_current()
                    return
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    if (
                        sink_generation != self._sink_generation
                        or stream is not self._current
                        or (sink is not None and sink is not self._sink)
                    ):
                        return
                    reason = str(exc) if str(exc).startswith("audio_") else "audio_output_failed"
                    self._fail_current(reason)
            self._sync_activity()

        def _drain_current(self, *, schedule: bool = True) -> None:
            stream = self._current
            sink = self._sink
            write_device = self._write_device
            if stream is None or sink is None or write_device is None:
                return
            if self._drain_in_progress:
                self._drain_requested = True
                return
            sink_generation = self._sink_generation

            def still_owned() -> bool:
                return bool(
                    sink_generation == self._sink_generation
                    and stream is self._current
                    and sink is self._sink
                    and write_device is self._write_device
                )

            self._drain_in_progress = True
            try:
                try:
                    while stream.pending_bytes > 0:
                        bytes_free = getattr(sink, "bytesFree", None)
                        available = (
                            int(bytes_free()) if callable(bytes_free) else stream.pending_bytes
                        )
                        if not still_owned():
                            return
                        if available <= 0:
                            break
                        payload = stream.peek(min(available, 256 * 1024))
                        if not payload:
                            break
                        written = int(write_device.write(payload))
                        if not still_owned():
                            return
                        if written < 0 or written > len(payload):
                            self._fail_current("audio_output_write_failed")
                            return
                        if written == 0:
                            break
                        self._stalled_drains = 0
                        stream.consume(written)
                        self._queued_bytes = max(0, self._queued_bytes - written)
                        self._written_bytes += written
                        self._write_generation += 1
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    if not still_owned():
                        return
                    self._fail_current("audio_output_write_failed")
                    return
            finally:
                self._drain_in_progress = False
                if self._active_state_seen and still_owned():
                    # ActiveState 可能发生在 write() 中；此时外层 write 已经
                    # 完成，必须把新增的写入代际纳入 active 回执。
                    self._active_write_generation = max(
                        self._active_write_generation,
                        self._write_generation,
                    )
                if self._drain_requested and still_owned():
                    self._drain_requested = False
                    self._drain_current(schedule=False)
            if stream.pending_bytes > 0 and schedule:
                self._schedule_drain()
            elif stream.final:
                self._schedule_completion_check()

        def _schedule_drain(self) -> None:
            if self._closed or self._drain_scheduled:
                return
            self._drain_scheduled = True
            sink_generation = self._sink_generation

            def drain() -> None:
                if sink_generation != self._sink_generation:
                    return
                self._drain_scheduled = False
                stream = self._current
                before = stream.pending_bytes if stream is not None else 0
                self._drain_current(schedule=False)
                current = self._current
                if current is None or sink_generation != self._sink_generation:
                    return
                if current.pending_bytes < before:
                    self._stalled_drains = 0
                elif current.pending_bytes > 0:
                    self._stalled_drains += 1
                if self._stalled_drains >= self._max_stalled_drains:
                    self._fail_current("audio_output_stalled")
                    return
                if current.pending_bytes > 0:
                    self._schedule_drain()

            try:
                self._drain_scheduler(drain)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                self._drain_scheduled = False
                self._fail_current("audio_output_schedule_failed")

        def _schedule_completion_check(self) -> None:
            if self._closed or self._completion_scheduled or self._current is None:
                return
            self._completion_scheduled = True
            sink_generation = self._sink_generation

            def check() -> None:
                if sink_generation != self._sink_generation:
                    return
                self._completion_scheduled = False
                stream = self._current
                sink = self._sink
                if stream is None or sink is None or not stream.final or stream.pending_bytes:
                    return
                try:
                    state = sink.state() if callable(getattr(sink, "state", None)) else None
                    if (
                        sink_generation != self._sink_generation
                        or stream is not self._current
                        or sink is not self._sink
                    ):
                        return
                    audio_state = getattr(QAudio, "State", QAudio)
                    idle_state = getattr(audio_state, "IdleState", object())
                    if state == idle_state:
                        processed = (
                            int(sink.processedUSecs())
                            if callable(getattr(sink, "processedUSecs", None))
                            else 0
                        )
                        if (
                            sink_generation != self._sink_generation
                            or stream is not self._current
                            or sink is not self._sink
                        ):
                            return
                        frame_bytes = max(1, stream.channels * 2)
                        expected = int(
                            self._written_bytes
                            * 1_000_000
                            / max(1, stream.sample_rate * frame_bytes)
                        )
                        drained = expected > 0 and processed + 1000 >= expected
                        if drained:
                            self._idle_write_generation = self._write_generation
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    if (
                        sink_generation != self._sink_generation
                        or stream is not self._current
                        or sink is not self._sink
                    ):
                        return
                    self._fail_current("audio_output_state_failed")
                    return
                self._maybe_finish_current()
                if self._current is None:
                    return
                self._completion_polls += 1
                if self._completion_polls >= self._max_completion_polls:
                    self._fail_current("audio_output_completion_timeout")
                    return
                self._schedule_completion_check()

            try:
                self._drain_scheduler(check)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                self._completion_scheduled = False
                self._fail_current("audio_output_schedule_failed")

        def _maybe_finish_current(self) -> None:
            stream = self._current
            if (
                stream is not None
                and stream.final
                and stream.pending_bytes == 0
                and self._write_generation > 0
                and self._idle_write_generation >= self._write_generation
            ):
                self._finish_current()

        def _fail_current(self, reason_code: str) -> None:
            stream = self._current
            if stream is None:
                return
            self._dropped_streams += 1
            self._last_error = str(reason_code or "audio_output_failed")[:64]
            with self._cancelled_lock:
                self._cancelled_contexts.add(stream.context)
            log_event(
                logger,
                "audio.playback.failed",
                component="gui.qt6.audio",
                status="failed",
                level=logging.WARNING,
                correlation_id=stream.context,
                reason_code=self._last_error,
            )
            self._finish_current(
                discard=True,
                terminal_state="failed",
                reason_code=self._last_error,
            )

        def _emit_receipt(
            self,
            stream: _PlaybackStream,
            state: str,
            *,
            reason_code: str = "",
        ) -> None:
            try:
                elapsed = (
                    max(0.0, (monotonic() - self._stream_started_at) * 1000.0)
                    if self._stream_started_at > 0
                    else 0.0
                )
                self.playbackReceipt.emit(
                    PlaybackReceipt(
                        context=stream.context,
                        request_id=stream.request_id,
                        state=str(state),
                        bytes_written=max(0, int(self._written_bytes)),
                        duration_ms=elapsed,
                        sample_rate=stream.sample_rate,
                        channels=stream.channels,
                        reason_code=str(reason_code or "")[:64],
                    )
                )
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("failed to emit audio playback receipt", exc_info=True)

        def _finish_current(
            self,
            *,
            discard: bool = False,
            terminal_state: str = "completed",
            reason_code: str = "",
        ) -> None:
            stream = self._current
            sink = self._sink
            pending_bytes = stream.pending_bytes if stream is not None else 0
            written_bytes = max(0, int(self._written_bytes))
            started_at = self._stream_started_at
            receipt_duration_ms = (
                max(0.0, (monotonic() - started_at) * 1000.0) if started_at > 0 else 0.0
            )
            self._queued_bytes = max(0, self._queued_bytes - pending_bytes)
            if stream is not None:
                try:
                    self.playbackReceipt.emit(
                        PlaybackReceipt(
                            context=stream.context,
                            request_id=stream.request_id,
                            state=str(terminal_state),
                            bytes_written=written_bytes,
                            duration_ms=receipt_duration_ms,
                            sample_rate=stream.sample_rate,
                            channels=stream.channels,
                            reason_code=str(reason_code or "")[:64],
                        )
                    )
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to emit audio playback receipt", exc_info=True)
            self._current = None
            self._sink = None
            self._write_device = None
            self._current_context = None
            self._drain_scheduled = False
            self._completion_scheduled = False
            self._sink_generation += 1
            self._write_generation = 0
            self._active_write_generation = 0
            self._idle_write_generation = 0
            self._written_bytes = 0
            self._drain_in_progress = False
            self._drain_requested = False
            self._active_state_seen = False
            self._stream_started_at = 0.0
            self._active_device_id = ""
            self._stalled_drains = 0
            self._completion_polls = 0
            if sink is not None:
                try:
                    reset = getattr(sink, "reset", None)
                    stop = getattr(sink, "stop", None)
                    if discard and callable(reset):
                        reset()
                    elif callable(stop):
                        stop()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
                delete_later = getattr(sink, "deleteLater", None)
                if callable(delete_later):
                    delete_later()
            self._sync_activity()
            self._play_next()

        def _sync_activity(self) -> None:
            active = self.playing and not self._closed
            if active == self._activity:
                return
            self._activity = active
            self.playbackStateChanged.emit(active)

        def cancel(self, context: object) -> None:
            """丢弃指定回合的排队、当前和设备缓冲音频。"""

            if self._closed:
                return
            with self._cancelled_lock:
                self._cancelled_contexts.add(context)
                if len(self._cancelled_contexts) > 1024:
                    self._cancelled_contexts.pop()
            self.cancelRequested.emit(context)

        def _is_cancelled(self, context: object) -> bool:
            with self._cancelled_lock:
                return context in self._cancelled_contexts

        def clear_cancelled(self, context: object) -> None:
            with self._cancelled_lock:
                self._cancelled_contexts.discard(context)

        def _cancel_context(self, context: object) -> None:
            retained: deque[_PlaybackStream] = deque()
            for stream in self._pending:
                if stream.context == context:
                    self._queued_bytes = max(0, self._queued_bytes - stream.pending_bytes)
                else:
                    retained.append(stream)
            self._pending = retained
            if self._current is not None and self._current.context == context:
                self._finish_current(
                    discard=True,
                    terminal_state="cancelled",
                    reason_code="cancelled",
                )
            else:
                self._sync_activity()

        def stop_all(self) -> None:
            """线程安全地停止所有物理播放；不把上下文永久标记为取消。"""

            if not self._closed:
                with self._epoch_lock:
                    self._playback_epoch += 1
                self.stopAllRequested.emit()

        def _stop_all(self) -> None:
            self._pending.clear()
            self._queued_bytes = 0
            if self._current is not None:
                self._finish_current(
                    discard=True,
                    terminal_state="cancelled",
                    reason_code="stopped",
                )
            else:
                self._sync_activity()

        def _state_changed(self, sink: object, generation: int, state: Any) -> None:
            if (
                self._closed
                or sink is not self._sink
                or generation != self._sink_generation
                or self._current is None
            ):
                return
            audio_state = getattr(QAudio, "State", QAudio)
            active_state = getattr(audio_state, "ActiveState", object())
            idle_state = getattr(audio_state, "IdleState", object())
            stopped_state = getattr(audio_state, "StoppedState", object())
            if state == active_state:
                self._active_state_seen = True
                self._drain_current()
                self._active_write_generation = self._write_generation
                return
            if state == idle_state:
                # 先清除上一轮 Active 标记；如果 drain 过程中重新收到
                # ActiveState，说明设备仍在播放，不能提前完成该流。
                self._active_state_seen = False
                self._drain_current()
                if not self._active_state_seen:
                    self._idle_write_generation = max(
                        self._idle_write_generation,
                        self._active_write_generation,
                    )
                self._maybe_finish_current()
                return
            if state == stopped_state:
                error = getattr(self._sink, "error", None)
                error_value = error() if callable(error) else None
                no_error = getattr(getattr(QAudio, "Error", QAudio), "NoError", None)
                current = self._current
                if error_value != no_error or (
                    current is not None
                    and (
                        current.pending_bytes > 0
                        or not current.final
                        or self._write_generation == 0
                        or self._idle_write_generation < self._write_generation
                    )
                ):
                    self._fail_current("audio_output_stopped")
                else:
                    self._finish_current(discard=True, terminal_state="completed")

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            self._disconnect_device_listener()
            with self._epoch_lock:
                self._playback_epoch += 1
            self._pending.clear()
            self._queued_bytes = 0
            with self._cancelled_lock:
                self._cancelled_contexts.clear()
            sink = self._sink
            stream = self._current
            written_bytes = max(0, int(self._written_bytes))
            receipt_duration_ms = (
                max(0.0, (monotonic() - self._stream_started_at) * 1000.0)
                if self._stream_started_at > 0
                else 0.0
            )
            if stream is not None:
                try:
                    self.playbackReceipt.emit(
                        PlaybackReceipt(
                            context=stream.context,
                            request_id=stream.request_id,
                            state="cancelled",
                            bytes_written=written_bytes,
                            duration_ms=receipt_duration_ms,
                            sample_rate=stream.sample_rate,
                            channels=stream.channels,
                            reason_code="closed",
                        )
                    )
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to emit audio playback receipt", exc_info=True)
            self._current = None
            self._sink = None
            self._write_device = None
            self._current_context = None
            self._drain_scheduled = False
            self._completion_scheduled = False
            self._sink_generation += 1
            if sink is not None:
                try:
                    reset = getattr(sink, "reset", None)
                    if callable(reset):
                        reset()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
                delete_later = getattr(sink, "deleteLater", None)
                if callable(delete_later):
                    delete_later()
            self._sync_activity()

else:

    class _NoopSignal:
        """无 Qt 时维持播放回执接口的空信号。"""

        def connect(self, _callback: object) -> None:
            return None

        def emit(self, _value: object) -> None:
            return None

    class QtAudioPlayer:  # pragma: no cover - 仅在无 Qt 环境使用
        """无 Qt Multimedia 时的明确不可用占位。"""

        available = False
        playing = False

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.playbackReceipt = _NoopSignal()
            return None

        def __call__(self, _chunk: AudioChunk) -> None:
            return None

        def cancel(self, _context: object) -> None:
            return None

        def clear_cancelled(self, _context: object) -> None:
            return None

        def stop_all(self) -> None:
            return None

        def diagnostics(self) -> dict[str, object]:
            return {
                "status": "unavailable",
                "available": False,
                "active": False,
                "queued_streams": 0,
                "queued_bytes": 0,
                "dropped_streams": 0,
                "error_code": "qt_multimedia_unavailable",
            }

        @property
        def output_device_id(self) -> str:
            return ""

        def set_output_device_id(self, _device_id: object) -> str:
            return ""

        def output_devices(self) -> tuple[AudioOutputDeviceChoice, ...]:
            return ()

        def close(self) -> None:
            return None


__all__ = ["AudioOutputDeviceChoice", "PlaybackReceipt", "QtAudioPlayer"]
