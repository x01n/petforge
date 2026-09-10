from __future__ import annotations

import asyncio
import copy
import os
import struct
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QByteArray, QObject, Signal
    from PySide6.QtMultimedia import QAudio, QAudioFormat
    from PySide6.QtWidgets import QApplication
except (ImportError, ModuleNotFoundError, OSError):  # pragma: no cover
    pytest.skip("PySide6 Qt Multimedia is unavailable", allow_module_level=True)

from app.runtime import asr_capture_configuration, build_runtime
from config.loader import ConfigurationError, LoadedConfiguration, default_configuration_values
from config.resources import inspect_resources
from gui.qt6.app import _close_qt_interaction_resources
from gui.qt6.config_panel import _FIELD_SPECS, AudioInputSelector, ConfigurationPanel
from gui.qt6.console import PetConsoleWindow
from gui.qt6.microphone import (
    AudioInputDeviceChoice,
    CapturedPcm,
    QtMicrophoneCapture,
    QtPushToTalkController,
)


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


class _Signal:
    def __init__(self) -> None:
        self.callbacks: list[object] = []

    def connect(self, callback: object) -> None:
        self.callbacks.append(callback)

    def emit(self, *values: object) -> None:
        for callback in tuple(self.callbacks):
            callback(*values)  # type: ignore[operator]


class _Timer:
    def __init__(self, _parent: object) -> None:
        self.timeout = _Signal()
        self.started_ms = 0
        self.running = False

    def setSingleShot(self, _enabled: bool) -> None:  # noqa: N802
        return None

    def start(self, milliseconds: int) -> None:
        self.started_ms = int(milliseconds)
        self.running = True

    def stop(self) -> None:
        self.running = False

    def fire(self) -> None:
        self.running = False
        self.timeout.emit()


class _IODevice:
    def __init__(self) -> None:
        self.readyRead = _Signal()
        self.payloads: list[bytes] = []

    def push(self, data: bytes) -> None:
        self.payloads.append(bytes(data))
        self.readyRead.emit()

    def readAll(self) -> QByteArray:  # noqa: N802
        value = b"".join(self.payloads)
        self.payloads.clear()
        return QByteArray(value)


class _Device:
    def __init__(
        self,
        identifier: str,
        *,
        target_supported: bool = True,
        preferred: QAudioFormat | None = None,
        description: str | None = None,
    ) -> None:
        self.identifier = identifier
        self.target_supported = target_supported
        self._preferred = preferred or _format(48_000, 2, QAudioFormat.SampleFormat.Int16)
        self._description = description or identifier

    def id(self) -> QByteArray:
        return QByteArray(self.identifier.encode("utf-8"))

    def isNull(self) -> bool:  # noqa: N802
        return False

    def description(self) -> str:
        return self._description

    def isFormatSupported(self, value: QAudioFormat) -> bool:  # noqa: N802
        if (
            value.sampleRate() == 16_000
            and value.channelCount() == 1
            and value.sampleFormat() == QAudioFormat.SampleFormat.Int16
        ):
            return self.target_supported
        return (
            value.sampleRate() == self._preferred.sampleRate()
            and value.channelCount() == self._preferred.channelCount()
            and value.sampleFormat() == self._preferred.sampleFormat()
        )

    def preferredFormat(self) -> QAudioFormat:  # noqa: N802
        return self._preferred


class _MediaDevices:
    def __init__(self, devices: list[_Device], default: _Device | None = None) -> None:
        self.devices = devices
        self.default = default if default is not None else (devices[0] if devices else None)
        self.audioInputsChanged = _Signal()
        self.audio_inputs_calls = 0

    def audioInputs(self) -> list[_Device]:  # noqa: N802
        self.audio_inputs_calls += 1
        return list(self.devices)

    def defaultAudioInput(self) -> _Device | None:  # noqa: N802
        return self.default


class _Source:
    def __init__(self, io_device: _IODevice | None = None) -> None:
        self.stateChanged = _Signal()
        self.io_device = io_device or _IODevice()
        self.stopped = False
        self.start_count = 0
        self.audio_error = QAudio.Error.NoError

    def start(self) -> _IODevice:
        self.start_count += 1
        return self.io_device

    def stop(self) -> None:
        self.stopped = True

    def error(self) -> object:
        return self.audio_error


class _Player(QObject):
    playbackStateChanged = Signal(bool)

    def __init__(self, *, playing: bool = False) -> None:
        super().__init__()
        self.playing = playing
        self.stop_count = 0

    def stop_all(self) -> None:
        self.stop_count += 1
        self.playing = False
        self.playbackStateChanged.emit(False)


class _Permission:
    def __init__(self, status: object) -> None:
        self._status = status

    def status(self) -> object:
        return self._status


def _format(
    sample_rate: int,
    channels: int,
    sample_format: QAudioFormat.SampleFormat,
) -> QAudioFormat:
    value = QAudioFormat()
    value.setSampleRate(sample_rate)
    value.setChannelCount(channels)
    value.setSampleFormat(sample_format)
    return value


def _capture(
    *,
    permission: object = "granted",
    requester: object | None = None,
    device: _Device | None = None,
    device_id: str = "",
    max_audio_bytes: int = 4096,
    duration: float = 15.0,
    source: _Source | None = None,
) -> tuple[QtMicrophoneCapture, _Source, _Timer]:
    selected = device or _Device("microphone-a")
    selected_source = source or _Source()
    timer: _Timer | None = None

    def timer_factory(parent: object) -> _Timer:
        nonlocal timer
        timer = _Timer(parent)
        return timer

    capture = QtMicrophoneCapture(
        enabled=True,
        max_audio_bytes=max_audio_bytes,
        max_duration_seconds=duration,
        device_id=device_id,
        permission_checker=lambda: permission,
        permission_requester=requester,  # type: ignore[arg-type]
        media_devices=_MediaDevices([selected]),
        audio_source_factory=lambda _device, _format, _parent: selected_source,
        timer_factory=timer_factory,
    )
    assert timer is not None
    return capture, selected_source, timer


def test_microphone_permission_is_only_checked_after_explicit_request(
    application: QApplication,
) -> None:
    del application
    checks: list[str] = []
    callbacks: list[object] = []
    capture, source, _timer = _capture(
        permission="undetermined",
        requester=lambda callback: callbacks.append(callback),
    )
    capture._permission_checker = lambda: checks.append("checked") or "undetermined"
    assert checks == []
    assert source.stopped is False

    assert capture.request_start()["status"] == "permission_pending"
    assert checks == ["checked"]
    assert capture.state == "permission_pending"
    callbacks[0](_Permission("granted"))  # type: ignore[operator]
    assert capture.recording is True
    capture.close()

    denied, denied_source, _timer = _capture(permission="denied")
    assert denied.request_start()["status"] == "denied"
    assert denied.state == "denied"
    assert denied_source.stopped is False
    denied.close()

    late_callbacks: list[object] = []
    late, late_source, _timer = _capture(
        permission="undetermined",
        requester=lambda callback: late_callbacks.append(callback),
    )
    late.request_start()
    late.close()
    late_callbacks[0]("granted")  # type: ignore[operator]
    assert late.state == "closed"
    assert late_source.stopped is False


def test_permission_result_applies_latest_capture_configuration_before_start(
    application: QApplication,
) -> None:
    del application
    granted_callbacks: list[object] = []
    capture, source, _timer = _capture(
        permission="undetermined",
        requester=lambda callback: granted_callbacks.append(callback),
    )
    capture.request_start()
    capture.configure(
        enabled=False,
        max_audio_bytes=1024,
        max_duration_seconds=2.0,
        device_id="",
    )
    granted_callbacks[0](_Permission("granted"))  # type: ignore[operator]
    assert capture.state == "disabled"
    assert capture.public_status()["enabled"] is False
    assert capture._pending_configuration is None
    assert source.start_count == 0
    capture.close()

    denied_callbacks: list[object] = []
    denied, denied_source, _timer = _capture(
        permission="undetermined",
        requester=lambda callback: denied_callbacks.append(callback),
    )
    denied.request_start()
    denied.configure(
        enabled=False,
        max_audio_bytes=1024,
        max_duration_seconds=2.0,
        device_id="",
    )
    denied_callbacks[0](_Permission("denied"))  # type: ignore[operator]
    assert denied.state == "disabled"
    assert denied.public_status()["enabled"] is False
    assert denied._pending_configuration is None
    assert denied_source.start_count == 0
    denied.close()


def test_permission_callback_generation_is_single_use_and_preserves_recording(
    application: QApplication,
) -> None:
    del application
    callbacks: list[object] = []
    capture, source, _timer = _capture(
        permission="undetermined",
        requester=lambda callback: callbacks.append(callback),
    )
    assert capture.request_start()["status"] == "permission_pending"
    callback = callbacks[0]
    callback(_Permission("granted"))  # type: ignore[operator]
    assert capture.state == "recording"
    assert source.start_count == 1
    callback(_Permission("granted"))  # type: ignore[operator]
    callback(_Permission("denied"))  # type: ignore[operator]
    assert capture.state == "recording"
    assert capture.recording is True
    assert source.start_count == 1
    assert source.stopped is False
    capture.close()
    assert source.stopped is True
    callback(_Permission("denied"))  # type: ignore[operator]
    assert capture.state == "closed"
    assert source.stopped is True

    denied_callbacks: list[object] = []
    denied, denied_source, _timer = _capture(
        permission="undetermined",
        requester=lambda callback: denied_callbacks.append(callback),
    )
    denied.request_start()
    denied_callback = denied_callbacks[0]
    denied_callback(_Permission("denied"))  # type: ignore[operator]
    denied_callback(_Permission("granted"))  # type: ignore[operator]
    assert denied.state == "denied"
    assert denied.recording is False
    assert denied_source.start_count == 0
    assert denied_source.stopped is False
    denied.close()

    synchronous, sync_source, _timer = _capture(
        permission="undetermined",
        requester=lambda callback: callback(_Permission("granted")),
    )
    assert synchronous.request_start()["status"] == "recording"
    assert sync_source.start_count == 1
    synchronous.close()


def test_permission_grant_keeps_half_duplex_gate_until_reconfigured_capture_starts(
    application: QApplication,
) -> None:
    del application
    callbacks: list[object] = []
    capture, source, timer = _capture(
        permission="undetermined",
        requester=lambda callback: callbacks.append(callback),
    )
    states: list[str] = []
    capture.stateChanged.connect(lambda state, _message: states.append(state))
    capture.request_start()
    capture.configure(
        enabled=True,
        max_audio_bytes=1024,
        max_duration_seconds=2.0,
        device_id="",
    )
    callbacks[0](_Permission("granted"))  # type: ignore[operator]

    assert states == ["permission_pending", "recording"]
    assert capture.recording is True
    assert timer.started_ms == 2000
    assert source.start_count == 1
    capture.close()


def test_microphone_uses_exact_device_id_and_rejects_missing_device(
    application: QApplication,
) -> None:
    del application
    first = _Device("first")
    selected = _Device("selected")
    source = _Source()
    seen: list[object] = []
    capture = QtMicrophoneCapture(
        enabled=True,
        max_audio_bytes=4096,
        max_duration_seconds=15,
        device_id="selected",
        permission_checker=lambda: "granted",
        media_devices=_MediaDevices([first, selected], default=first),
        audio_source_factory=lambda device, _format, _parent: seen.append(device) or source,
        timer_factory=_Timer,
    )
    assert capture.request_start()["status"] == "recording"
    assert seen == [selected]
    assert "selected" not in repr(capture.public_status())
    capture.close()

    missing = QtMicrophoneCapture(
        enabled=True,
        max_audio_bytes=4096,
        max_duration_seconds=15,
        device_id="missing",
        permission_checker=lambda: "granted",
        media_devices=_MediaDevices([first]),
        audio_source_factory=lambda _device, _format, _parent: _Source(),
        timer_factory=_Timer,
    )
    result = missing.request_start()
    assert result == {"status": "unavailable", "reason_code": "device_missing"}
    assert missing.state == "unavailable"
    missing.close()


def test_microphone_device_listing_is_lazy_private_and_recovers_after_hotplug(
    application: QApplication,
) -> None:
    del application
    first = _Device("stable-id", description="桌面麦克风")
    replacement = _Device("replacement-id", description="桌面麦克风")
    media = _MediaDevices([first], default=first)
    permission_checks: list[str] = []
    capture = QtMicrophoneCapture(
        enabled=True,
        max_audio_bytes=4096,
        max_duration_seconds=15,
        device_id="stable-id",
        permission_checker=lambda: permission_checks.append("checked") or "granted",
        media_devices=media,
        audio_source_factory=lambda _device, _format, _parent: _Source(),
        timer_factory=_Timer,
    )
    assert media.audio_inputs_calls == 0
    assert capture.input_devices_loaded is False
    choices = capture.load_input_devices()
    assert media.audio_inputs_calls == 1
    assert permission_checks == []
    assert [choice.description for choice in choices] == ["桌面麦克风"]
    assert choices[0].device_id == "stable-id"
    assert "stable-id" not in repr(choices[0])
    assert "stable-id" not in repr(capture.public_status())

    media.devices = [replacement]
    media.default = replacement
    media.audioInputsChanged.emit()
    assert capture.state == "unavailable"
    assert "不可用" in capture._message

    media.devices = [first, replacement]
    media.default = first
    media.audioInputsChanged.emit()
    assert capture.state == "idle"
    assert [choice.description for choice in capture.input_device_choices] == [
        "桌面麦克风（1）",
        "桌面麦克风（2）",
    ]
    capture.close()


def test_audio_input_selector_keeps_default_empty_and_hides_device_ids(
    application: QApplication,
) -> None:
    selector = AudioInputSelector()
    choices = (
        AudioInputDeviceChoice("private-a", "USB 麦克风", is_default=True),
        AudioInputDeviceChoice("private-b", "耳机麦克风"),
    )
    selector.set_devices(choices, loaded=True, selected_device_id="")
    assert selector.selected_device_id() == ""
    assert selector.itemText(0).startswith("跟随系统默认")
    assert "private-a" not in selector.itemText(1)
    assert selector.itemData(1) == "private-a"
    selector.set_selected_device_id("missing-private-id")
    assert selector.currentText() == "已选设备暂不可用"
    assert selector.selected_device_id() == "missing-private-id"
    selector.deleteLater()


def test_microphone_limit_and_timer_stop_emit_bounded_pcm(application: QApplication) -> None:
    del application
    capture, source, _timer = _capture(max_audio_bytes=1024)
    delivered: list[CapturedPcm] = []
    capture.captureReady.connect(delivered.append)
    assert capture.request_start()["status"] == "recording"
    source.io_device.push(b"\x01\x00" * 700)
    assert capture.state == "captured"
    assert len(delivered) == 1
    assert len(delivered[0].data) == 1024
    assert delivered[0].stop_reason == "limit"
    assert source.stopped is True
    capture.close()

    timed, timed_source, timer = _capture(max_audio_bytes=4096, duration=1.25)
    timed_result: list[CapturedPcm] = []
    timed.captureReady.connect(timed_result.append)
    timed.request_start()
    assert timer.started_ms == 1250
    timed_source.io_device.push(b"\x02\x00" * 32)
    timer.fire()
    assert timed_result[0].stop_reason == "duration"
    timed.close()


def test_microphone_stop_drains_pcm_buffered_before_ready_read(
    application: QApplication,
) -> None:
    del application
    capture, source, _timer = _capture(max_audio_bytes=4096)
    delivered: list[CapturedPcm] = []
    capture.captureReady.connect(delivered.append)
    capture.request_start()
    source.io_device.payloads.append(b"\x03\x00" * 48)

    result = capture.stop()
    assert result["status"] == "captured"
    assert delivered[0].data == b"\x03\x00" * 48
    capture.close()


def test_microphone_converts_supported_float_fallback_to_s16le(
    application: QApplication,
) -> None:
    del application
    preferred = _format(48_000, 2, QAudioFormat.SampleFormat.Float)
    device = _Device("float-device", target_supported=False, preferred=preferred)
    capture, source, _timer = _capture(device=device)
    delivered: list[CapturedPcm] = []
    capture.captureReady.connect(delivered.append)
    assert capture.request_start()["status"] == "recording"
    source.io_device.push(struct.pack("<ffff", -1.0, 0.0, 0.5, 1.0))
    capture.stop()
    assert delivered[0].sample_rate == 48_000
    assert delivered[0].channels == 2
    assert struct.unpack("<hhhh", delivered[0].data) == (-32768, 0, 16384, 32767)
    capture.close()


def test_microphone_device_loss_discards_private_audio(application: QApplication) -> None:
    del application
    capture, source, _timer = _capture()
    delivered: list[CapturedPcm] = []
    capture.captureReady.connect(delivered.append)
    capture.request_start()
    source.io_device.push(b"\x01\x00" * 16)
    source.audio_error = QAudio.Error.IOError
    source.stateChanged.emit(QAudio.State.StoppedState)
    assert capture.state == "unavailable"
    assert delivered == []
    assert capture.public_status() == {
        "state": "unavailable",
        "enabled": True,
        "recording": False,
    }
    capture.close()


def test_microphone_hot_reload_waits_for_active_capture_boundary(
    application: QApplication,
) -> None:
    del application
    capture, source, timer = _capture(max_audio_bytes=4096, duration=10.0)
    delivered: list[CapturedPcm] = []
    capture.captureReady.connect(delivered.append)
    capture.request_start()
    capture.configure(
        enabled=False,
        max_audio_bytes=1024,
        max_duration_seconds=2.0,
        device_id="next-device",
    )
    assert capture.recording is True
    assert timer.started_ms == 10_000
    source.io_device.push(b"\x01\x00" * 32)
    capture.stop()
    assert len(delivered) == 1
    assert capture.state == "disabled"
    assert capture.public_status()["enabled"] is False
    capture.close()


def test_ptt_controller_stops_playback_and_only_emits_completed_text(
    application: QApplication,
) -> None:
    capture, source, _timer = _capture()
    player = _Player(playing=True)
    future: Future[object] = Future()
    submitted: list[CapturedPcm] = []
    controller = QtPushToTalkController(
        capture,
        transcription_submitter=lambda value: submitted.append(value) or future,
        audio_player=player,
    )
    states: list[str] = []
    text: list[str] = []
    controller.stateChanged.connect(lambda state, _message: states.append(state))
    controller.transcriptionReady.connect(text.append)

    assert controller.toggle()["status"] == "waiting_playback"
    application.processEvents()
    assert player.stop_count == 1
    assert capture.recording is True
    assert controller.blocks_playback is True
    source.io_device.push(b"\x01\x00" * 32)
    assert controller.toggle()["status"] == "captured"
    assert controller.state == "transcribing"
    assert len(submitted) == 1
    future.set_result(SimpleNamespace(text="识别结果"))
    controller._poll_transcription()
    assert text == ["识别结果"]
    assert controller.state == "completed"
    assert controller.blocks_playback is False
    assert "recording" in states and "transcribing" in states
    controller.close()


def test_ptt_controller_close_cancels_transcription_future(application: QApplication) -> None:
    capture, source, _timer = _capture()
    future: Future[object] = Future()
    controller = QtPushToTalkController(
        capture,
        transcription_submitter=lambda _value: future,
        audio_player=_Player(),
    )
    assert controller.toggle()["status"] == "waiting_playback"
    application.processEvents()
    assert capture.recording is True
    source.io_device.push(b"\x01\x00" * 32)
    controller.toggle()
    assert controller.transcribing is True
    controller.close()
    assert future.cancelled() is True
    assert controller.state == "closed"


def test_app_shutdown_closes_active_ptt_before_audio_and_ignores_late_result(
    application: QApplication,
) -> None:
    capture, source, _timer = _capture()
    future: Future[object] = Future()
    events: list[str] = []

    class _TrackedPlayer(_Player):
        def close(self) -> None:
            events.append("audio_close")

    class _TrackedController(QtPushToTalkController):
        def close(self) -> None:
            events.append("controller_close")
            super().close()

    player = _TrackedPlayer()
    controller = _TrackedController(
        capture,
        transcription_submitter=lambda _value: future,
        audio_player=player,
    )
    controller.toggle()
    application.processEvents()
    source.io_device.push(b"\x01\x00" * 32)
    controller.toggle()
    assert controller.transcribing is True
    _close_qt_interaction_resources(controller, player)
    assert events == ["controller_close", "audio_close"]
    assert source.stopped is True
    assert future.cancelled() is True
    late_text: list[str] = []
    controller.transcriptionReady.connect(late_text.append)
    controller._poll_transcription()
    assert late_text == []
    _close_qt_interaction_resources(controller, player)
    assert events == ["controller_close", "audio_close", "controller_close", "audio_close"]


def test_ptt_completion_keeps_capture_disabled_by_recording_boundary_reload(
    application: QApplication,
) -> None:
    capture, source, _timer = _capture()
    future: Future[object] = Future()
    controller = QtPushToTalkController(
        capture,
        transcription_submitter=lambda _value: future,
        audio_player=_Player(),
    )
    text: list[str] = []
    controller.transcriptionReady.connect(text.append)
    controller.toggle()
    application.processEvents()
    capture.configure(
        enabled=False,
        max_audio_bytes=1024,
        max_duration_seconds=2.0,
        device_id="",
    )
    source.io_device.push(b"\x01\x00" * 32)
    controller.toggle()
    assert controller.state == "transcribing"
    assert capture.public_status()["enabled"] is False

    future.set_result(SimpleNamespace(text="识别结果"))
    controller._poll_transcription()
    assert text == ["识别结果"]
    assert controller.state == "disabled"
    assert controller.blocks_playback is False
    controller.close()


@pytest.mark.parametrize("outcome", ["error", "empty"])
def test_ptt_failure_keeps_capture_disabled_by_recording_boundary_reload(
    application: QApplication,
    outcome: str,
) -> None:
    capture, source, _timer = _capture()
    future: Future[object] = Future()
    controller = QtPushToTalkController(
        capture,
        transcription_submitter=lambda _value: future,
        audio_player=_Player(),
    )
    controller.toggle()
    application.processEvents()
    capture.configure(
        enabled=False,
        max_audio_bytes=1024,
        max_duration_seconds=2.0,
        device_id="",
    )
    source.io_device.push(b"\x01\x00" * 32)
    controller.toggle()
    assert capture.public_status()["enabled"] is False
    if outcome == "error":
        future.set_exception(RuntimeError("private failure"))
    else:
        future.set_result(SimpleNamespace(text=""))
    controller._poll_transcription()

    assert controller.state == "disabled"
    assert controller.blocks_playback is False
    controller.close()


def test_console_microphone_only_fills_input_and_reflows(application: QApplication) -> None:
    calls: list[str] = []
    submissions: list[str] = []
    window = PetConsoleWindow(
        callbacks={
            "microphone_toggle": lambda: calls.append("toggle") or {"status": "requested"},
            "submit": submissions.append,
        }
    )
    window.setMinimumSize(1, 1)
    window.show()
    window._console_tabs.setCurrentIndex(window._conversation_page_index)
    window.set_microphone_state("recording")
    for width, columns in ((680, 4), (420, 3), (180, 1)):
        window.resize(width, 500)
        application.processEvents()
        assert window._conversation_input_columns == columns
        for name in (
            "conversationInput",
            "microphoneToggleButton",
            "sendMessageButton",
            "stopConversationButton",
            "microphoneStatusLabel",
        ):
            child = window.findChild(QObject, name)
            assert child is not None and child.isVisible()
            right = child.mapTo(window, child.rect().bottomRight()).x()
            assert right <= window.rect().right()
    window.set_microphone_state("idle")
    window.microphone_button.click()
    assert calls == ["toggle"]
    window.set_microphone_state("recording")
    assert window.microphone_button.text() == "停止并识别"
    window.microphone_button.click()
    assert calls == ["toggle", "toggle"]
    window.input_line.setText("已有草稿")
    output = window.output_view.toPlainText()
    window.set_microphone_transcription("识别\n结果")
    assert window.input_line.text() == "已有草稿 识别 结果"
    assert submissions == []
    assert window.output_view.toPlainText() == output
    window.shutdown()
    application.processEvents()


def test_console_microphone_shutdown_ignores_late_projection(application: QApplication) -> None:
    window = PetConsoleWindow(callbacks={"microphone_toggle": lambda: None})
    window.input_line.setText("保留")
    window.set_microphone_state("transcribing")
    window.shutdown()
    application.processEvents()
    window.set_microphone_state("completed")
    window.set_microphone_transcription("迟到正文")
    assert window._microphone_state == "closed"
    assert window.microphone_button.isEnabled() is False
    assert window.input_line.text() == "保留"


def test_asr_capture_configuration_is_strict_and_exposed_in_config_panel() -> None:
    assert asr_capture_configuration({}) == {
        "enabled": True,
        "max_duration_seconds": 15.0,
        "auto_submit": False,
        "device_id": "",
    }
    with pytest.raises(ConfigurationError, match="auto_submit"):
        asr_capture_configuration({"capture": {"auto_submit": True}})
    with pytest.raises(ConfigurationError, match="device_id"):
        asr_capture_configuration({"capture": {"device_id": 7}})
    paths = {spec.path for spec in _FIELD_SPECS["asr"]}
    assert {
        ("capture", "enabled"),
        ("capture", "max_duration_seconds"),
        ("capture", "auto_submit"),
        ("capture", "device_id"),
    } <= paths


def test_asr_capture_controls_are_editable_without_raw_yaml(application: QApplication) -> None:
    panel = ConfigurationPanel(
        {
            "asr": {
                "enabled": False,
                "backend": "sensevoice",
                "language": "zh",
                "capture": {
                    "enabled": True,
                    "max_duration_seconds": 15.0,
                    "auto_submit": False,
                    "device_id": "",
                },
            }
        },
        developer_mode=False,
    )
    assert panel.open_section("asr") is True
    application.processEvents()
    controls = panel._friendly_controls["asr"]
    assert ("capture", "enabled") in controls
    assert ("capture", "max_duration_seconds") in controls
    assert ("capture", "device_id") in controls
    assert controls[("capture", "auto_submit")].isEnabled() is False
    panel.close()
    application.processEvents()


def test_capture_only_runtime_reload_does_not_replace_asr_worker(tmp_path: Path) -> None:
    values = default_configuration_values(
        resource_root=tmp_path / "resources",
        database_path=tmp_path / "meapet.sqlite3",
    )
    values["tts"] = {"enabled": False, "backend": "text_only", "language": "zh"}
    values["asr"]["enabled"] = False
    initial = LoadedConfiguration(tmp_path / "app.yaml", values)
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    original_asr = runtime.asr
    runtime.asr.has_active_tasks = lambda: True  # type: ignore[method-assign]
    updated_values = copy.deepcopy(values)
    updated_values["asr"]["capture"]["max_duration_seconds"] = 20.0
    updated = LoadedConfiguration(tmp_path / "app.yaml", updated_values)

    async def scenario() -> None:
        result = await runtime.apply_configuration(updated)
        assert result["status"] == "reloaded"
        assert result["applied_sections"] == ("asr",)
        assert runtime.asr is original_asr
        assert runtime.configuration.values["asr"]["capture"]["max_duration_seconds"] == 20.0
        await runtime.close()

    asyncio.run(scenario())
