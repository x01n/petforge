from __future__ import annotations

from threading import Thread

from PySide6.QtCore import QCoreApplication
from PySide6.QtMultimedia import QAudio

from core.events.types import AudioChunk, ConversationContext
from gui.qt6.audio import QtAudioPlayer


class _Signal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, value) -> None:
        for callback in tuple(self._callbacks):
            callback(value)


class _Device:
    def __init__(
        self,
        *,
        supported: bool = True,
        device_id: bytes = b"",
        description: str = "test speaker",
    ) -> None:
        self.supported = supported
        self.device_id = device_id
        self.device_description = description

    def isNull(self) -> bool:  # noqa: N802 - Qt 测试替身
        return False

    def isFormatSupported(self, _audio_format) -> bool:  # noqa: N802
        return self.supported

    def id(self) -> bytes:
        return self.device_id

    def description(self) -> str:
        return self.device_description


class _MediaDevices:
    def __init__(self, outputs: list[_Device], default: _Device | None = None) -> None:
        self.audioOutputsChanged = _Signal()
        self.outputs = outputs
        self.default = default or (outputs[0] if outputs else None)

    def audioOutputs(self) -> tuple[_Device, ...]:  # noqa: N802
        return tuple(self.outputs)

    def defaultAudioOutput(self) -> _Device | None:  # noqa: N802
        return self.default


class _WriteDevice:
    def __init__(
        self,
        maximum_write: int | None = None,
        return_value: int | None = None,
        on_write=None,
        raise_after_callback: bool = False,
    ) -> None:
        self.maximum_write = maximum_write
        self.return_value = return_value
        self.on_write = on_write
        self.raise_after_callback = raise_after_callback
        self.writes: list[bytes] = []

    def write(self, value: bytes) -> int:
        data = bytes(value)
        if self.on_write is not None:
            self.on_write()
        if self.raise_after_callback:
            raise RuntimeError("write failed after state callback")
        if self.return_value is not None:
            if self.return_value > 0:
                data = data[: self.return_value]
                self.writes.append(data)
            return self.return_value
        if self.maximum_write is not None:
            data = data[: self.maximum_write]
        self.writes.append(data)
        return len(data)


class _Sink:
    def __init__(
        self,
        *,
        bytes_free: int = 1024 * 1024,
        maximum_write: int | None = None,
        return_value: int | None = None,
        start_states: tuple[object, ...] = (),
        start_returns_none: bool = False,
    ) -> None:
        self.stateChanged = _Signal()
        self.bytes_free = bytes_free
        self.writer = _WriteDevice(maximum_write, return_value)
        self.start_states = start_states
        self.start_returns_none = start_returns_none
        self._state = QAudio.State.IdleState
        self.processed_usecs = 10_000_000
        self.on_state = None
        self.raise_state = False
        self.reset_count = 0
        self.stop_count = 0
        self.deleted = False

    def start(self):
        for state in self.start_states:
            self.emit_state(state)
        if self.start_returns_none:
            return None
        return self.writer

    def bytesFree(self) -> int:  # noqa: N802
        return self.bytes_free

    def error(self):
        return QAudio.Error.NoError

    def state(self):
        callback = self.on_state
        self.on_state = None
        if callback is not None:
            callback()
        if self.raise_state:
            raise RuntimeError("state failed after callback")
        return self._state

    def bufferSize(self) -> int:  # noqa: N802
        return self.bytes_free if self.bytes_free > 0 else 1024

    def processedUSecs(self) -> int:  # noqa: N802
        return self.processed_usecs

    def reset(self) -> None:
        self.reset_count += 1

    def stop(self) -> None:
        self.stop_count += 1

    def deleteLater(self) -> None:  # noqa: N802
        self.deleted = True

    def emit_state(self, state) -> None:
        self._state = state
        self.stateChanged.emit(state)


class _SinkFactory:
    def __init__(
        self,
        *,
        bytes_free: int = 1024 * 1024,
        maximum_write: int | None = None,
        return_value: int | None = None,
        start_states: tuple[object, ...] = (),
    ) -> None:
        self.bytes_free = bytes_free
        self.maximum_write = maximum_write
        self.return_value = return_value
        self.start_states = start_states
        self.sinks: list[_Sink] = []
        self.devices: list[object] = []

    def __call__(self, device, _audio_format, _owner) -> _Sink:
        sink = _Sink(
            bytes_free=self.bytes_free,
            maximum_write=self.maximum_write,
            return_value=self.return_value,
            start_states=self.start_states,
        )
        self.sinks.append(sink)
        self.devices.append(device)
        return sink


class _SequenceSinkFactory:
    def __init__(
        self,
        start_states: tuple[tuple[object, ...], ...],
        *,
        start_returns_none: frozenset[int] = frozenset(),
        bytes_free: int = 1024 * 1024,
    ) -> None:
        self.start_states = start_states
        self.start_returns_none = start_returns_none
        self.bytes_free = bytes_free
        self.sinks: list[_Sink] = []

    def __call__(self, _device, _audio_format, _owner) -> _Sink:
        index = len(self.sinks)
        states = self.start_states[index] if index < len(self.start_states) else ()
        sink = _Sink(
            bytes_free=self.bytes_free,
            start_states=states,
            start_returns_none=index in self.start_returns_none,
        )
        self.sinks.append(sink)
        return sink


def _context(turn: str = "turn") -> ConversationContext:
    return ConversationContext("profile", "session", turn, 1)


def _chunk(
    context: ConversationContext,
    data: bytes,
    *,
    final: bool = False,
    request_id: str = "request",
    sample_rate: int = 32000,
) -> AudioChunk:
    return AudioChunk(
        context,
        data=data,
        sample_rate=sample_rate,
        channels=1,
        is_final=final,
        request_id=request_id,
    )


def _app() -> QCoreApplication:
    return QCoreApplication.instance() or QCoreApplication([])


def test_player_reuses_one_sink_for_all_chunks_until_explicit_final() -> None:
    _app()
    factory = _SinkFactory()
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )
    states: list[bool] = []
    player.playbackStateChanged.connect(states.append)
    context = _context()

    player(_chunk(context, b"first"))
    player(_chunk(context, b"second"))
    player(_chunk(context, b"", final=True))

    assert len(factory.sinks) == 1
    assert b"".join(factory.sinks[0].writer.writes) == b"firstsecond"
    assert player.playing is True
    factory.sinks[0].emit_state(QAudio.State.ActiveState)
    factory.sinks[0].emit_state(QAudio.State.IdleState)
    assert player.playing is False
    assert states == [True, False]
    assert player.diagnostics()["dropped_streams"] == 0
    player.close()


def test_player_resolves_exact_output_device_and_lists_descriptions() -> None:
    _app()
    first = _Device(device_id=b"speaker-a", description="扬声器 A")
    second = _Device(device_id=b"speaker-b", description="扬声器 B")
    media = _MediaDevices([first, second], default=first)
    factory = _SinkFactory()
    player = QtAudioPlayer(
        media_devices=media,
        output_device_id="speaker-b",
        sink_factory=factory,
    )

    choices = player.output_devices()
    assert [choice.device_id for choice in choices] == ["speaker-a", "speaker-b"]
    assert [choice.description for choice in choices] == ["扬声器 A", "扬声器 B"]
    assert choices[0].is_default is True
    assert player.output_device_id == "speaker-b"

    player(_chunk(_context("selected"), b"audio", final=True))
    assert factory.devices == [second]
    player.close()


def test_output_device_removal_fails_active_stream_with_receipt() -> None:
    _app()
    first = _Device(device_id=b"speaker-a")
    media = _MediaDevices([first], default=first)
    factory = _SinkFactory(bytes_free=0)
    player = QtAudioPlayer(
        media_devices=media,
        sink_factory=factory,
        drain_scheduler=lambda _callback: None,
    )
    receipts = []
    player.playbackReceipt.connect(receipts.append)
    player(_chunk(_context("removed"), b"audio", request_id="removed"))
    assert player.playing is True

    media.outputs.clear()
    media.default = None
    media.audioOutputsChanged.emit(None)

    assert player.playing is False
    assert player.diagnostics()["error_code"] == "audio_output_disconnected"
    assert receipts[-1].state == "failed"
    assert receipts[-1].reason_code == "audio_output_disconnected"
    player.close()


def test_default_device_change_keeps_stream_when_active_device_remains() -> None:
    _app()
    first = _Device(device_id=b"speaker-a")
    second = _Device(device_id=b"speaker-b")
    media = _MediaDevices([first, second], default=first)
    factory = _SinkFactory(bytes_free=0)
    player = QtAudioPlayer(
        media_devices=media,
        sink_factory=factory,
        drain_scheduler=lambda _callback: None,
    )
    player(_chunk(_context("default-change"), b"audio", request_id="default-change"))
    media.default = second
    media.audioOutputsChanged.emit(None)

    assert player.playing is True
    assert player.diagnostics()["error_code"] == ""
    player.close()


def test_player_finishes_when_empty_final_arrives_after_device_became_idle() -> None:
    _app()
    factory = _SinkFactory()
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )
    context = _context()

    player(_chunk(context, b"sentence"))
    factory.sinks[0].emit_state(QAudio.State.ActiveState)
    factory.sinks[0].emit_state(QAudio.State.IdleState)
    assert player.playing is True

    player(_chunk(context, b"", final=True))
    assert player.playing is False
    assert factory.sinks[0].stop_count == 1
    player.close()


def test_player_does_not_silently_drop_more_than_sixty_four_chunks() -> None:
    _app()
    factory = _SinkFactory()
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )
    context = _context()

    for index in range(100):
        player(_chunk(context, bytes([index])))
    player(_chunk(context, b"", final=True))

    assert len(factory.sinks) == 1
    assert b"".join(factory.sinks[0].writer.writes) == bytes(range(100))
    assert player.diagnostics()["dropped_streams"] == 0
    factory.sinks[0].emit_state(QAudio.State.ActiveState)
    factory.sinks[0].emit_state(QAudio.State.IdleState)
    player.close()


def test_player_serializes_different_contexts_without_interleaving() -> None:
    _app()
    factory = _SinkFactory()
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )

    player(_chunk(_context("one"), b"one", final=True, request_id="one"))
    player(_chunk(_context("two"), b"two", final=True, request_id="two"))
    assert len(factory.sinks) == 1
    factory.sinks[0].emit_state(QAudio.State.ActiveState)
    factory.sinks[0].emit_state(QAudio.State.IdleState)

    assert len(factory.sinks) == 2
    assert factory.sinks[0].writer.writes == [b"one"]
    assert factory.sinks[1].writer.writes == [b"two"]
    factory.sinks[1].emit_state(QAudio.State.ActiveState)
    factory.sinks[1].emit_state(QAudio.State.IdleState)
    assert player.playing is False
    player.close()


def test_player_merges_interleaved_pending_chunks_by_request_key() -> None:
    _app()
    factory = _SinkFactory()
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )
    first = _context("first")
    second = _context("second")
    third = _context("third")

    player(_chunk(first, b"A", final=True, request_id="A"))
    player(_chunk(second, b"B1", request_id="B"))
    player(_chunk(third, b"C", final=True, request_id="C"))
    player(_chunk(second, b"B2", final=True, request_id="B"))

    assert [stream.request_id for stream in player._pending] == ["B", "C"]
    assert player._pending[0].final is True
    factory.sinks[0].emit_state(QAudio.State.ActiveState)
    factory.sinks[0].emit_state(QAudio.State.IdleState)
    assert b"".join(factory.sinks[1].writer.writes) == b"B1B2"
    factory.sinks[1].emit_state(QAudio.State.ActiveState)
    factory.sinks[1].emit_state(QAudio.State.IdleState)
    assert factory.sinks[2].writer.writes == [b"C"]
    factory.sinks[2].emit_state(QAudio.State.ActiveState)
    factory.sinks[2].emit_state(QAudio.State.IdleState)
    assert player.playing is False
    player.close()


def test_late_state_from_old_sink_cannot_reset_new_stream() -> None:
    _app()
    factory = _SinkFactory()
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )
    player(_chunk(_context("old"), b"old", final=True, request_id="old"))
    player(_chunk(_context("new"), b"new", final=True, request_id="new"))
    old_sink = factory.sinks[0]
    old_sink.emit_state(QAudio.State.ActiveState)
    old_sink.emit_state(QAudio.State.IdleState)
    new_sink = factory.sinks[1]

    old_sink.emit_state(QAudio.State.StoppedState)
    assert player.playing is True
    assert new_sink.reset_count == 0
    assert player.diagnostics()["dropped_streams"] == 0
    new_sink.emit_state(QAudio.State.ActiveState)
    new_sink.emit_state(QAudio.State.IdleState)
    player.close()


def test_sync_stopped_during_start_cannot_overwrite_recursively_started_stream() -> None:
    _app()
    factory = _SequenceSinkFactory(
        (
            (),
            (QAudio.State.StoppedState,),
            (),
        )
    )
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )
    first = _context("first")
    broken = _context("broken")
    current = _context("current")
    player(_chunk(first, b"A", final=True, request_id="A"))
    player(_chunk(broken, b"B", final=True, request_id="B"))
    player(_chunk(current, b"C1", request_id="C"))
    factory.sinks[0].emit_state(QAudio.State.ActiveState)
    factory.sinks[0].emit_state(QAudio.State.IdleState)

    assert len(factory.sinks) == 3
    player(_chunk(current, b"C2", final=True, request_id="C"))
    assert factory.sinks[1].writer.writes == []
    assert b"".join(factory.sinks[2].writer.writes) == b"C1C2"
    factory.sinks[2].emit_state(QAudio.State.ActiveState)
    factory.sinks[2].emit_state(QAudio.State.IdleState)
    assert player.playing is False
    player.close()


def test_sync_stopped_then_start_failure_cannot_fail_recursively_started_stream() -> None:
    _app()
    factory = _SequenceSinkFactory(
        (
            (),
            (QAudio.State.StoppedState,),
            (),
        ),
        start_returns_none=frozenset({1}),
    )
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )
    first = _context("first")
    broken = _context("broken")
    current = _context("current")
    player(_chunk(first, b"A", final=True, request_id="A"))
    player(_chunk(broken, b"B", final=True, request_id="B"))
    player(_chunk(current, b"C", final=True, request_id="C"))
    factory.sinks[0].emit_state(QAudio.State.ActiveState)
    factory.sinks[0].emit_state(QAudio.State.IdleState)

    assert len(factory.sinks) == 3
    assert player.playing is True
    assert player._current is not None and player._current.request_id == "C"
    assert factory.sinks[2].reset_count == 0
    factory.sinks[2].emit_state(QAudio.State.ActiveState)
    factory.sinks[2].emit_state(QAudio.State.IdleState)
    assert player.playing is False
    player.close()


def test_write_state_reentry_cannot_consume_or_count_new_stream() -> None:
    _app()
    scheduled = []
    factory = _SequenceSinkFactory(((), ()), bytes_free=0)
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=scheduled.append,
    )
    first = _context("first")
    current = _context("current")
    player(_chunk(first, b"AAAA", request_id="A"))
    player(_chunk(current, b"BBBB", final=True, request_id="B"))
    factory.sinks[0].bytes_free = 1024
    factory.sinks[0].writer.on_write = lambda: factory.sinks[0].emit_state(
        QAudio.State.StoppedState
    )

    scheduled.pop(0)()
    assert player._current is not None and player._current.request_id == "B"
    assert player._current.pending_bytes == 4
    assert player.diagnostics()["queued_bytes"] == 4
    assert player._write_generation == 0
    player.close()


def test_active_state_reentry_during_write_does_not_duplicate_pcm_or_stall() -> None:
    _app()
    scheduled = []
    factory = _SinkFactory(bytes_free=0)
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=scheduled.append,
    )
    receipts = []
    player.playbackReceipt.connect(receipts.append)
    context = _context("active-reentry")

    player(_chunk(context, b"audio", final=True))
    sink = factory.sinks[0]
    sink.bytes_free = 1024
    sink.writer.on_write = lambda: sink.emit_state(QAudio.State.ActiveState)

    assert len(scheduled) == 1
    scheduled.pop(0)()

    assert sink.writer.writes == [b"audio"]
    assert player.playing is True
    sink.emit_state(QAudio.State.IdleState)

    assert player.playing is False
    assert [receipt.state for receipt in receipts] == ["started", "completed"]
    assert receipts[-1].bytes_written == len(b"audio")
    assert receipts[-1].as_dict()["request_id"] == "request"
    assert "context" not in receipts[-1].as_dict()
    player.close()


def test_write_exception_after_state_reentry_cannot_fail_new_stream() -> None:
    _app()
    scheduled = []
    factory = _SequenceSinkFactory(((), ()), bytes_free=0)
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=scheduled.append,
    )
    first = _context("first")
    current = _context("current")
    player(_chunk(first, b"AAAA", request_id="A"))
    player(_chunk(current, b"BBBB", final=True, request_id="B"))
    factory.sinks[0].bytes_free = 1024
    factory.sinks[0].writer.on_write = lambda: factory.sinks[0].emit_state(
        QAudio.State.StoppedState
    )
    factory.sinks[0].writer.raise_after_callback = True

    scheduled.pop(0)()
    assert player._current is not None and player._current.request_id == "B"
    assert player._current.pending_bytes == 4
    assert factory.sinks[1].reset_count == 0
    assert player.diagnostics()["dropped_streams"] == 1
    player.close()


def test_player_drains_partial_writes_without_restarting_sink() -> None:
    _app()
    scheduled = []
    factory = _SinkFactory(bytes_free=2, maximum_write=2)
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=scheduled.append,
    )
    context = _context()
    player(_chunk(context, b"abcdef", final=True))

    while scheduled:
        scheduled.pop(0)()

    assert len(factory.sinks) == 1
    assert b"".join(factory.sinks[0].writer.writes) == b"abcdef"
    factory.sinks[0].emit_state(QAudio.State.ActiveState)
    factory.sinks[0].emit_state(QAudio.State.IdleState)
    player.close()


def test_new_data_after_idle_requires_a_new_active_idle_cycle_before_final() -> None:
    _app()
    factory = _SinkFactory()
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )
    context = _context()
    player(_chunk(context, b"first"))
    factory.sinks[0].emit_state(QAudio.State.ActiveState)
    factory.sinks[0].emit_state(QAudio.State.IdleState)

    player(_chunk(context, b"second"))
    player(_chunk(context, b"", final=True))
    assert player.playing is True
    assert factory.sinks[0].stop_count == 0

    factory.sinks[0].emit_state(QAudio.State.ActiveState)
    factory.sinks[0].emit_state(QAudio.State.IdleState)
    assert player.playing is False
    assert b"".join(factory.sinks[0].writer.writes) == b"firstsecond"
    player.close()


def test_sync_start_state_reentry_reaches_terminal_via_completion_probe() -> None:
    _app()
    scheduled = []
    factory = _SinkFactory(
        start_states=(QAudio.State.ActiveState, QAudio.State.IdleState),
    )
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=scheduled.append,
    )
    player(_chunk(_context(), b"audio", final=True))
    assert player.playing is True
    assert scheduled

    scheduled.pop(0)()
    assert player.playing is False
    assert factory.sinks[0].writer.writes == [b"audio"]
    player.close()


def test_negative_write_and_repeated_zero_write_have_error_terminals() -> None:
    _app()
    negative_factory = _SinkFactory(return_value=-1)
    negative = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=negative_factory,
    )
    negative(_chunk(_context("negative"), b"audio", final=True))
    assert negative.playing is False
    assert negative.diagnostics()["error_code"] == "audio_output_write_failed"
    negative.close()

    scheduled = []
    zero_factory = _SinkFactory(return_value=0)
    stalled = QtAudioPlayer(
        max_stalled_drains=3,
        output_device_provider=lambda: _Device(),
        sink_factory=zero_factory,
        drain_scheduler=scheduled.append,
    )
    stalled(_chunk(_context("zero"), b"audio", final=True))
    while scheduled:
        scheduled.pop(0)()
    assert stalled.playing is False
    assert stalled.diagnostics()["error_code"] == "audio_output_stalled"
    stalled.close()

    no_capacity_callbacks = []
    no_capacity_factory = _SinkFactory(bytes_free=0)
    no_capacity = QtAudioPlayer(
        max_stalled_drains=3,
        output_device_provider=lambda: _Device(),
        sink_factory=no_capacity_factory,
        drain_scheduler=no_capacity_callbacks.append,
    )
    no_capacity(_chunk(_context("capacity"), b"audio", final=True))
    while no_capacity_callbacks:
        no_capacity_callbacks.pop(0)()
    assert no_capacity.playing is False
    assert no_capacity.diagnostics()["error_code"] == "audio_output_stalled"
    no_capacity.close()


def test_write_larger_than_payload_is_rejected_as_audio_output_failure() -> None:
    _app()
    factory = _SinkFactory(return_value=1024)
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )

    player(_chunk(_context("oversized-write"), b"audio", final=True))

    assert player.playing is False
    assert player.diagnostics()["dropped_streams"] == 1
    assert player.diagnostics()["error_code"] == "audio_output_write_failed"
    player.close()


def test_producer_burst_does_not_consume_timer_based_stall_budget() -> None:
    _app()
    scheduled = []
    factory = _SinkFactory(bytes_free=0)
    player = QtAudioPlayer(
        max_stalled_drains=3,
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=scheduled.append,
    )
    context = _context()
    player(_chunk(context, b"one"))
    player(_chunk(context, b"two"))
    player(_chunk(context, b"three", final=True))

    assert player.playing is True
    assert player.diagnostics()["error_code"] == ""
    assert len(scheduled) == 1
    while scheduled:
        scheduled.pop(0)()
    assert player.playing is False
    assert player.diagnostics()["error_code"] == "audio_output_stalled"
    player.close()


def test_player_cancel_and_stop_all_reset_device_without_poisoning_new_stream() -> None:
    _app()
    factory = _SinkFactory(bytes_free=0)
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=lambda _callback: None,
    )
    context = _context()
    player(_chunk(context, b"queued"))
    player.cancel(context)
    assert factory.sinks[0].reset_count == 1
    assert player.playing is False

    player.clear_cancelled(context)
    factory.bytes_free = 1024
    player(_chunk(context, b"new", final=True, request_id="new"))
    assert len(factory.sinks) == 2
    player.stop_all()
    assert factory.sinks[1].reset_count == 1
    assert player.playing is False
    player.close()


def test_stop_all_rejects_audio_signal_queued_from_an_older_epoch() -> None:
    app = _app()
    factory = _SinkFactory()
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )
    chunk = _chunk(_context(), b"queued", final=True)

    worker = Thread(target=lambda: player(chunk))
    worker.start()
    worker.join()
    player.stop_all()
    app.processEvents()

    assert factory.sinks == []
    assert player.playing is False
    player.close()


def test_stopped_without_error_is_failure_when_stream_is_incomplete() -> None:
    _app()
    factory = _SinkFactory(bytes_free=0)
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=lambda _callback: None,
    )
    player(_chunk(_context(), b"pending"))
    factory.sinks[0].emit_state(QAudio.State.StoppedState)

    assert player.playing is False
    assert player.diagnostics()["dropped_streams"] == 1
    assert player.diagnostics()["error_code"] == "audio_output_stopped"
    player.close()


def test_stopped_without_idle_confirmation_rejects_buffered_final_audio() -> None:
    _app()
    factory = _SinkFactory()
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
    )
    player(_chunk(_context(), b"buffered", final=True))
    assert player._write_generation == 1
    assert player._idle_write_generation == 0
    factory.sinks[0].emit_state(QAudio.State.StoppedState)

    assert player.playing is False
    assert player.diagnostics()["dropped_streams"] == 1
    assert player.diagnostics()["error_code"] == "audio_output_stopped"
    player.close()


def test_close_resets_and_deletes_active_sink() -> None:
    _app()
    factory = _SinkFactory(bytes_free=0)
    player = QtAudioPlayer(
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=lambda _callback: None,
    )
    player(_chunk(_context(), b"pending"))
    player.close()

    assert factory.sinks[0].reset_count == 1
    assert factory.sinks[0].deleted is True
    assert player.playing is False


def test_player_queue_overflow_is_explicit_and_cancels_corrupted_context() -> None:
    _app()
    factory = _SinkFactory(bytes_free=0)
    player = QtAudioPlayer(
        max_pending_bytes=4096,
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=lambda _callback: None,
    )
    context = _context()
    player(_chunk(context, b"a" * 4000))
    player(_chunk(context, b"b" * 200))

    diagnostics = player.diagnostics()
    assert diagnostics["dropped_streams"] == 1
    assert diagnostics["error_code"] == "queue_limit"
    assert diagnostics["queued_bytes"] == 0
    assert player.playing is False
    assert factory.sinks[0].reset_count == 1
    player.close()


def test_stream_limit_counts_every_discarded_stream_for_one_context() -> None:
    _app()
    factory = _SinkFactory(bytes_free=0)
    player = QtAudioPlayer(
        max_pending_streams=2,
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=lambda _callback: None,
    )
    context = _context()
    player(_chunk(context, b"A", request_id="A"))
    player(_chunk(context, b"B", request_id="B"))
    player(_chunk(context, b"C", final=True, request_id="C"))

    assert player.playing is False
    assert player.diagnostics()["error_code"] == "stream_limit"
    assert player.diagnostics()["dropped_streams"] == 2
    assert player.diagnostics()["queued_streams"] == 0
    player.close()


def test_completion_poll_has_bounded_timeout_when_sink_never_becomes_idle() -> None:
    _app()
    scheduled = []
    factory = _SinkFactory()
    player = QtAudioPlayer(
        max_completion_polls=10,
        output_device_provider=lambda: _Device(),
        sink_factory=factory,
        drain_scheduler=scheduled.append,
    )
    player(_chunk(_context(), b"audio", final=True))
    factory.sinks[0]._state = QAudio.State.ActiveState

    while scheduled:
        scheduled.pop(0)()

    assert player.playing is False
    assert player.diagnostics()["error_code"] == "audio_output_completion_timeout"
    player.close()


def test_completion_state_reentry_cannot_finish_or_fail_new_stream() -> None:
    for raises in (False, True):
        _app()
        scheduled = []
        factory = _SinkFactory()
        player = QtAudioPlayer(
            output_device_provider=lambda: _Device(),
            sink_factory=factory,
            drain_scheduler=scheduled.append,
        )
        first = _context(f"first-{raises}")
        current = _context(f"current-{raises}")
        player(_chunk(first, b"AAAA", final=True, request_id="A"))
        player(_chunk(current, b"BBBB", final=True, request_id="B"))
        factory.sinks[0].on_state = lambda: factory.sinks[0].emit_state(QAudio.State.StoppedState)
        factory.sinks[0].raise_state = raises

        scheduled.pop(0)()
        assert player._current is not None and player._current.request_id == "B"
        assert player.playing is True
        assert factory.sinks[1].reset_count == 0
        assert factory.sinks[1].stop_count == 0
        assert player.diagnostics()["dropped_streams"] == 1
        player.close()


def test_player_reports_missing_or_unsupported_output_without_device_metadata() -> None:
    _app()
    missing = QtAudioPlayer(output_device_provider=lambda: None)
    assert missing.diagnostics() == {
        "status": "unavailable",
        "available": False,
        "active": False,
        "queued_streams": 0,
        "queued_bytes": 0,
        "dropped_streams": 0,
        "error_code": "",
    }
    missing(_chunk(_context(), b"audio", final=True))
    assert missing.diagnostics()["error_code"] == "audio_output_unavailable"
    assert missing.diagnostics()["dropped_streams"] == 1
    missing.close()

    unsupported = QtAudioPlayer(output_device_provider=lambda: _Device(supported=False))
    unsupported(_chunk(_context("unsupported"), b"audio", final=True))
    assert unsupported.diagnostics()["error_code"] == "audio_format_unsupported"
    unsupported.close()
