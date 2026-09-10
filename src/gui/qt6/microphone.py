from __future__ import annotations

import math
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

try:
    from PySide6.QtCore import QCoreApplication, QMicrophonePermission, QObject, Qt, QTimer, Signal

    _QT_CORE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):
    _QT_CORE_AVAILABLE = False

if _QT_CORE_AVAILABLE:
    try:
        from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSource, QMediaDevices

        _QT_MICROPHONE_AVAILABLE = True
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError):
        _QT_MICROPHONE_AVAILABLE = False
else:
    _QT_MICROPHONE_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class CapturedPcm:
    """一次仅驻留内存的 S16LE 录音。"""

    data: bytes
    sample_rate: int
    channels: int
    stop_reason: str = "user"

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("captured PCM must be non-empty bytes")
        if not 8_000 <= int(self.sample_rate) <= 192_000:
            raise ValueError("captured PCM sample rate is invalid")
        if not 1 <= int(self.channels) <= 8:
            raise ValueError("captured PCM channel count is invalid")
        if len(self.data) % (2 * int(self.channels)):
            raise ValueError("captured PCM is not frame aligned")


@dataclass(frozen=True, slots=True)
class AudioInputDeviceChoice:
    """仅供本地配置控件使用的输入设备选项。"""

    device_id: str = field(repr=False)
    description: str
    is_default: bool = False

    def __post_init__(self) -> None:
        device_id = str(self.device_id or "")
        if not device_id or len(device_id) > 1_024 or any(char in device_id for char in "\x00\r\n"):
            raise ValueError("audio input device id is invalid")
        description = " ".join(str(self.description or "").replace("\x00", " ").split())
        object.__setattr__(self, "device_id", device_id)
        object.__setattr__(self, "description", description[:160] or "未命名麦克风")


def _device_id_text(device: object) -> str:
    """把 Qt 设备 ID 转成可精确持久化的文本，不用于公开状态。"""

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
    """返回适合本地选择器显示的有界设备描述。"""

    description = getattr(device, "description", None)
    if not callable(description):
        return "未命名麦克风"
    try:
        rendered = " ".join(str(description() or "").replace("\x00", " ").split())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return "未命名麦克风"
    return rendered[:160] or "未命名麦克风"


def _permission_token(value: object) -> str:
    """归一化 Qt 权限枚举；字符串仅供依赖注入测试使用。"""

    status = getattr(value, "status", None)
    if callable(status):
        try:
            resolved = status()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            resolved = None
        if resolved is not None and resolved is not value:
            return _permission_token(resolved)
    name = str(getattr(value, "name", value) or "").strip().casefold()
    if name.endswith("granted"):
        return "granted"
    if name.endswith("denied"):
        return "denied"
    return "undetermined"


if _QT_MICROPHONE_AVAILABLE:

    class _PcmS16LeConverter:
        """把 Qt 明确报告的整型/浮点 PCM 转换为 S16LE，保留采样率和声道。"""

        _SAMPLE_BYTES = {"UInt8": 1, "Int16": 2, "Int32": 4, "Float": 4}

        def __init__(self, audio_format: QAudioFormat) -> None:
            self.sample_rate = int(audio_format.sampleRate())
            self.channels = int(audio_format.channelCount())
            sample_format = audio_format.sampleFormat()
            self.sample_format = str(getattr(sample_format, "name", sample_format))
            self.sample_bytes = self._SAMPLE_BYTES.get(self.sample_format, 0)
            if not 8_000 <= self.sample_rate <= 192_000:
                raise ValueError("microphone sample rate is unsupported")
            if not 1 <= self.channels <= 8:
                raise ValueError("microphone channel count is unsupported")
            if self.sample_bytes <= 0:
                raise ValueError("microphone sample format is unsupported")
            self.input_frame_bytes = self.sample_bytes * self.channels
            self.output_frame_bytes = 2 * self.channels
            self._carry = bytearray()

        def feed(self, data: bytes, *, output_limit: int) -> bytes:
            """转换完整帧并严格限制返回的 S16LE 字节数。"""

            if output_limit < self.output_frame_bytes:
                return b""
            combined = bytes(self._carry) + bytes(data)
            max_frames = output_limit // self.output_frame_bytes
            frame_count = min(len(combined) // self.input_frame_bytes, max_frames)
            consumed = frame_count * self.input_frame_bytes
            payload = combined[:consumed]
            self._carry = bytearray(combined[consumed : consumed + self.input_frame_bytes - 1])
            if not payload:
                return b""
            if self.sample_format == "Int16":
                if sys.byteorder == "little":
                    return payload
                values = struct.iter_unpack(">h", payload)
                return b"".join(struct.pack("<h", value[0]) for value in values)
            if self.sample_format == "UInt8":
                return b"".join(struct.pack("<h", (value - 128) << 8) for value in payload)
            if self.sample_format == "Int32":
                prefix = "<" if sys.byteorder == "little" else ">"
                values = struct.iter_unpack(f"{prefix}i", payload)
                return b"".join(struct.pack("<h", value[0] >> 16) for value in values)
            prefix = "<" if sys.byteorder == "little" else ">"
            converted = bytearray()
            for (value,) in struct.iter_unpack(f"{prefix}f", payload):
                if not math.isfinite(value):
                    value = 0.0
                value = max(-1.0, min(1.0, float(value)))
                sample = -32768 if value <= -1.0 else round(value * 32767.0)
                converted.extend(struct.pack("<h", sample))
            return bytes(converted)

    class QtMicrophoneCapture(QObject):
        """在 Qt 主线程采集有界 PCM；只有 ``request_start`` 会请求权限。"""

        stateChanged = Signal(str, str)
        captureReady = Signal(object)
        inputDevicesChanged = Signal(object)

        def __init__(
            self,
            *,
            enabled: bool = True,
            max_audio_bytes: int = 16 * 1024 * 1024,
            max_duration_seconds: float = 15.0,
            device_id: str = "",
            permission_checker: Callable[[], object] | None = None,
            permission_requester: Callable[[Callable[[object], None]], None] | None = None,
            media_devices: object | None = None,
            audio_source_factory: Callable[[object, QAudioFormat, QObject], object] | None = None,
            timer_factory: Callable[[QObject], object] | None = None,
            parent: QObject | None = None,
        ) -> None:
            super().__init__(parent)
            self._enabled = bool(enabled)
            self._max_audio_bytes = self._bounded_audio_limit(max_audio_bytes)
            self._max_duration_seconds = self._bounded_duration(max_duration_seconds)
            self._device_id = self._bounded_device_id(device_id)
            self._permission_checker = permission_checker or self._check_permission
            self._permission_requester = permission_requester or self._request_permission
            self._media_devices = media_devices or QMediaDevices(self)
            self._audio_source_factory = audio_source_factory or (
                lambda device, audio_format, parent: QAudioSource(device, audio_format, parent)
            )
            self._timer = (timer_factory or QTimer)(self)
            set_single_shot = getattr(self._timer, "setSingleShot", None)
            if callable(set_single_shot):
                set_single_shot(True)
            timeout = getattr(self._timer, "timeout", None)
            connect = getattr(timeout, "connect", None)
            if callable(connect):
                connect(self._auto_stop)
            self._state = "disabled" if not self._enabled else "idle"
            self._message = "语音输入已关闭" if not self._enabled else "语音输入待命"
            self._source: object | None = None
            self._io_device: object | None = None
            self._converter: _PcmS16LeConverter | None = None
            self._pcm = bytearray()
            self._closed = False
            self._stopping = False
            self._permission_generation = 0
            self._permission_consumed_generation: int | None = None
            self._pending_configuration: tuple[bool, int, float, str] | None = None
            self._input_device_choices: tuple[AudioInputDeviceChoice, ...] = ()
            self._devices_loaded = False
            self._device_listener_connected = False
            self._selected_device_missing = False
            self._active_device_id = ""

        @staticmethod
        def _bounded_audio_limit(value: object) -> int:
            if isinstance(value, bool):
                raise ValueError("microphone audio limit is invalid")
            limit = int(value)
            if not 1_024 <= limit <= 64 * 1024 * 1024:
                raise ValueError("microphone audio limit is invalid")
            return limit

        @staticmethod
        def _bounded_duration(value: object) -> float:
            if isinstance(value, bool):
                raise ValueError("microphone duration is invalid")
            duration = float(value)
            if not math.isfinite(duration) or not 0.1 <= duration <= 300.0:
                raise ValueError("microphone duration is invalid")
            return duration

        @staticmethod
        def _bounded_device_id(value: object) -> str:
            device_id = str(value or "")
            if len(device_id) > 1_024 or any(char in device_id for char in "\x00\r\n"):
                raise ValueError("microphone device id is invalid")
            return device_id

        @property
        def state(self) -> str:
            return self._state

        @property
        def recording(self) -> bool:
            return self._state == "recording"

        @property
        def busy(self) -> bool:
            return self._state in {"permission_pending", "recording"}

        def public_status(self) -> Mapping[str, object]:
            """公开状态不包含设备 ID、录音字节或正文。"""

            return {
                "state": self._state,
                "enabled": self._enabled,
                "recording": self.recording,
            }

        @property
        def input_devices_loaded(self) -> bool:
            """返回用户是否已经显式触发过设备枚举。"""

            return self._devices_loaded

        @property
        def input_device_choices(self) -> tuple[AudioInputDeviceChoice, ...]:
            """返回最近一次本地选择器快照；读取本属性不会枚举设备。"""

            return self._input_device_choices

        def _connect_device_listener(self) -> None:
            if self._device_listener_connected:
                return
            changed = getattr(self._media_devices, "audioInputsChanged", None)
            connect = getattr(changed, "connect", None)
            if callable(connect):
                connect(self._audio_inputs_changed)
                self._device_listener_connected = True

        def _device_snapshot(
            self,
        ) -> tuple[tuple[object, ...], object | None, tuple[AudioInputDeviceChoice, ...]]:
            inputs = getattr(self._media_devices, "audioInputs", None)
            default_input = getattr(self._media_devices, "defaultAudioInput", None)
            try:
                devices = tuple(inputs()) if callable(inputs) else ()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                devices = ()
            try:
                default_device = default_input() if callable(default_input) else None
                is_null = getattr(default_device, "isNull", None)
                if default_device is not None and callable(is_null) and bool(is_null()):
                    default_device = None
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                default_device = None
            default_id = _device_id_text(default_device) if default_device is not None else ""
            rows: list[tuple[str, str]] = []
            seen_ids: set[str] = set()
            description_counts: dict[str, int] = {}
            for device in devices:
                device_id = _device_id_text(device)
                if not device_id or device_id in seen_ids:
                    continue
                seen_ids.add(device_id)
                description = _device_description(device)
                rows.append((device_id, description))
                description_counts[description] = description_counts.get(description, 0) + 1
            description_indexes: dict[str, int] = {}
            choices: list[AudioInputDeviceChoice] = []
            for device_id, description in rows:
                if description_counts.get(description, 0) > 1:
                    index = description_indexes.get(description, 0) + 1
                    description_indexes[description] = index
                    description = f"{description}（{index}）"
                choices.append(
                    AudioInputDeviceChoice(
                        device_id=device_id,
                        description=description,
                        is_default=bool(default_id and device_id == default_id),
                    )
                )
            return devices, default_device, tuple(choices)

        def load_input_devices(self) -> tuple[AudioInputDeviceChoice, ...]:
            """响应用户展开或刷新选择器，不检查或请求麦克风权限。"""

            if self._closed:
                return ()
            self._connect_device_listener()
            devices, default_device, choices = self._device_snapshot()
            self._devices_loaded = True
            self._input_device_choices = choices
            self._reconcile_device_availability(devices, default_device)
            self.inputDevicesChanged.emit(choices)
            return choices

        def _audio_inputs_changed(self, *_args: object) -> None:
            if self._closed or not self._devices_loaded:
                return
            devices, default_device, choices = self._device_snapshot()
            self._input_device_choices = choices
            self._reconcile_device_availability(devices, default_device)
            self.inputDevicesChanged.emit(choices)

        def _reconcile_device_availability(
            self,
            devices: tuple[object, ...],
            default_device: object | None,
        ) -> None:
            del default_device
            available_ids = {_device_id_text(device) for device in devices}
            if self.recording and self._active_device_id not in available_ids:
                self._selected_device_missing = True
                self.abort("麦克风设备已断开，请重新选择后重试", reason_code="device_lost")
                return
            if not self._device_id:
                if self._selected_device_missing and not self.busy:
                    self._selected_device_missing = False
                    if self._enabled:
                        self._set_state("idle", "麦克风已恢复，可以重试")
                    else:
                        self._set_state("disabled", "语音输入已关闭")
                return
            missing = self._device_id not in available_ids
            if missing:
                self._selected_device_missing = True
                if self._state == "permission_pending":
                    self.abort("已选择的麦克风当前不可用，请重新连接后重试")
                elif self._enabled and not self.recording:
                    self._set_state(
                        "unavailable",
                        "已选择的麦克风当前不可用，请重新连接后重试",
                    )
                return
            if self._selected_device_missing:
                self._selected_device_missing = False
                if not self.busy:
                    self._set_state(
                        "idle" if self._enabled else "disabled",
                        "麦克风已恢复，可以重试" if self._enabled else "语音输入已关闭",
                    )

        def configure(
            self,
            *,
            enabled: bool,
            max_audio_bytes: int,
            max_duration_seconds: float,
            device_id: str = "",
        ) -> None:
            """空闲时原子更新后续录音参数；活动录音保持开始时快照。"""

            normalized_limit = self._bounded_audio_limit(max_audio_bytes)
            normalized_duration = self._bounded_duration(max_duration_seconds)
            normalized_device = self._bounded_device_id(device_id)
            normalized = (
                bool(enabled),
                normalized_limit,
                normalized_duration,
                normalized_device,
            )
            if self.busy:
                self._pending_configuration = normalized
                return
            self._apply_configuration(normalized)

        def _apply_configuration(
            self,
            values: tuple[bool, int, float, str],
            *,
            publish_state: bool = True,
        ) -> None:
            (
                self._enabled,
                self._max_audio_bytes,
                self._max_duration_seconds,
                self._device_id,
            ) = values
            self._pending_configuration = None
            if publish_state:
                self._set_state(
                    "idle" if self._enabled else "disabled",
                    "语音输入待命" if self._enabled else "语音输入已关闭",
                )
            if self._devices_loaded and not self.busy:
                self.load_input_devices()

        def _apply_pending_configuration(
            self,
            *,
            force: bool = False,
            publish_state: bool = True,
        ) -> None:
            pending = self._pending_configuration
            if pending is not None and (force or not self.busy):
                self._apply_configuration(pending, publish_state=publish_state)

        def _set_state(self, state: str, message: str) -> None:
            self._state = str(state)
            self._message = str(message)
            self.stateChanged.emit(self._state, self._message)

        def _check_permission(self) -> object:
            application = QCoreApplication.instance()
            if application is None:
                return Qt.PermissionStatus.Denied
            return application.checkPermission(QMicrophonePermission())

        def _request_permission(self, callback: Callable[[object], None]) -> None:
            application = QCoreApplication.instance()
            if application is None:
                callback(Qt.PermissionStatus.Denied)
                return
            application.requestPermission(QMicrophonePermission(), self, callback)

        def request_start(self) -> Mapping[str, object]:
            """响应用户点击；不会由启动、配置加载或后台任务隐式调用。"""

            if self._closed:
                return {"status": "closed"}
            if not self._enabled:
                self._set_state("disabled", "语音输入已关闭")
                return {"status": "disabled"}
            if self.recording:
                return {"status": "recording"}
            if self._state == "permission_pending":
                return {"status": "permission_pending"}
            self._permission_generation += 1
            generation = self._permission_generation
            try:
                permission = _permission_token(self._permission_checker())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                self._set_state("unavailable", "无法检查麦克风权限")
                return {"status": "unavailable", "reason_code": "permission_check_failed"}
            if permission == "granted":
                return self._start_capture()
            if permission == "denied":
                self._set_state("denied", "麦克风权限已被拒绝")
                return {"status": "denied", "reason_code": "permission_denied"}
            self._set_state("permission_pending", "等待麦克风权限确认…")

            def completed(value: object, selected_generation: int = generation) -> None:
                if (
                    self._closed
                    or selected_generation != self._permission_generation
                    or self._permission_consumed_generation == selected_generation
                ):
                    return
                # Qt 的权限回调在正常情况下只会执行一次；仍需在应用边界
                # 显式消费当前代次，避免重复 Granted/Denied 改写已启动录音。
                self._permission_consumed_generation = selected_generation
                # 权限弹窗期间配置可能已关闭录音或切换设备。先原子提交
                # 最新快照，迟到 Granted 不能按旧配置隐式启动麦克风。
                self._apply_pending_configuration(force=True, publish_state=False)
                if not self._enabled:
                    self._set_state("disabled", "语音输入已关闭")
                    return
                if _permission_token(value) == "granted":
                    self._start_capture()
                else:
                    self._set_state("denied", "麦克风权限已被拒绝")

            try:
                self._permission_requester(completed)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                # 某些测试替身/平台实现可能在回调已同步消费后再抛出；
                # 此时不得把已开始的录音重置为失败状态。
                if self._permission_consumed_generation == generation:
                    return {"status": self._state}
                self._set_state("unavailable", "无法请求麦克风权限")
                return {"status": "unavailable", "reason_code": "permission_request_failed"}
            if (
                self._permission_consumed_generation == generation
                and self._state != "permission_pending"
            ):
                return {"status": self._state}
            return {"status": "permission_pending"}

        def _selected_device(self) -> object | None:
            self._connect_device_listener()
            devices, default_device, choices = self._device_snapshot()
            self._devices_loaded = True
            self._input_device_choices = choices
            self.inputDevicesChanged.emit(choices)
            if self._device_id:
                selected = next(
                    (device for device in devices if _device_id_text(device) == self._device_id),
                    None,
                )
                self._selected_device_missing = selected is None
                return selected
            self._selected_device_missing = False
            return default_device

        @staticmethod
        def _selected_format(device: object) -> QAudioFormat | None:
            supported = getattr(device, "isFormatSupported", None)
            if not callable(supported):
                return None
            target = QAudioFormat()
            target.setSampleRate(16_000)
            target.setChannelCount(1)
            target.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            if supported(target):
                return target
            preferred = getattr(device, "preferredFormat", None)
            fallback = preferred() if callable(preferred) else None
            if not isinstance(fallback, QAudioFormat) or not supported(fallback):
                return None
            try:
                _PcmS16LeConverter(fallback)
            except ValueError:
                return None
            return fallback

        def _start_capture(self) -> Mapping[str, object]:
            if self._closed or not self._enabled:
                return {"status": "closed" if self._closed else "disabled"}
            device = self._selected_device()
            if device is None:
                self._set_state("unavailable", "没有可用的麦克风设备")
                return {"status": "unavailable", "reason_code": "device_missing"}
            audio_format = self._selected_format(device)
            if audio_format is None:
                self._set_state("unavailable", "麦克风格式不受支持")
                return {"status": "unavailable", "reason_code": "format_unsupported"}
            source: object | None = None
            try:
                converter = _PcmS16LeConverter(audio_format)
                source = self._audio_source_factory(device, audio_format, self)
                state_changed = getattr(source, "stateChanged", None)
                connect_state = getattr(state_changed, "connect", None)
                if callable(connect_state):
                    connect_state(self._source_state_changed)
                io_device = source.start()
                if io_device is None:
                    raise RuntimeError("audio source returned no IO device")
                error = getattr(source, "error", None)
                audio_error = getattr(QAudio, "Error", QAudio)
                no_error = getattr(audio_error, "NoError", object())
                if callable(error) and error() != no_error:
                    raise RuntimeError("audio source reported a startup error")
                ready_read = getattr(io_device, "readyRead", None)
                connect_read = getattr(ready_read, "connect", None)
                if not callable(connect_read):
                    raise RuntimeError("audio source IO device has no readyRead signal")
                connect_read(self._read_available)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                try:
                    if source is not None:
                        source.stop()
                except (AttributeError, RuntimeError):
                    pass
                self._set_state("unavailable", "麦克风启动失败")
                return {"status": "unavailable", "reason_code": "source_start_failed"}
            self._source = source
            self._io_device = io_device
            self._converter = converter
            self._active_device_id = _device_id_text(device)
            self._pcm.clear()
            self._stopping = False
            start_timer = getattr(self._timer, "start", None)
            if callable(start_timer):
                start_timer(max(1, round(self._max_duration_seconds * 1000)))
            self._set_state("recording", "录音中，再次点击后停止并识别")
            return {"status": "recording"}

        def _read_available(self, *, auto_stop: bool = True) -> None:
            if not self.recording or self._io_device is None or self._converter is None:
                return
            reader = getattr(self._io_device, "readAll", None)
            if not callable(reader):
                self.abort("麦克风读取失败", reason_code="read_failed")
                return
            try:
                raw = bytes(reader())
            except (OSError, RuntimeError, TypeError, ValueError):
                self.abort("麦克风读取失败", reason_code="read_failed")
                return
            remaining = self._max_audio_bytes - len(self._pcm)
            converted = self._converter.feed(raw, output_limit=max(0, remaining))
            if converted:
                self._pcm.extend(converted)
            capacity_reached = len(self._pcm) >= self._max_audio_bytes
            if not capacity_reached and raw:
                possible_frames = len(raw) // self._converter.input_frame_bytes
                converted_frames = len(converted) // self._converter.output_frame_bytes
                capacity_reached = possible_frames > converted_frames
            if capacity_reached and auto_stop:
                self.stop(reason="limit")

        def _source_state_changed(self, state: object) -> None:
            if not self.recording or self._stopping:
                return
            audio_state = getattr(QAudio, "State", QAudio)
            stopped = getattr(audio_state, "StoppedState", object())
            if state != stopped:
                return
            source = self._source
            error = getattr(source, "error", None)
            audio_error = getattr(QAudio, "Error", QAudio)
            no_error = getattr(audio_error, "NoError", object())
            try:
                source_error = error() if callable(error) else no_error
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                source_error = object()
            if source_error != no_error:
                self.abort("麦克风设备已断开", reason_code="device_lost")

        def _auto_stop(self) -> None:
            if self.recording:
                self.stop(reason="duration")

        def stop(self, *, reason: str = "user") -> Mapping[str, object]:
            """停止采集并把有界 PCM 只交给当前进程内的控制器。"""

            if not self.recording:
                return {"status": self._state}
            # QAudioSource 的 pull QIODevice 可能已有尚未触发下一次 readyRead
            # 的尾部数据；在 stop 使内部设备失效前冲刷一次。
            self._read_available(auto_stop=False)
            self._stopping = True
            stop_timer = getattr(self._timer, "stop", None)
            if callable(stop_timer):
                stop_timer()
            source = self._source
            if source is not None:
                try:
                    source.stop()
                except (AttributeError, RuntimeError):
                    pass
            converter = self._converter
            pcm = bytes(self._pcm)
            self._source = None
            self._io_device = None
            self._converter = None
            self._active_device_id = ""
            self._pcm.clear()
            self._stopping = False
            if not pcm or converter is None:
                self._set_state("unavailable", "没有采集到可识别的语音")
                return {"status": "unavailable", "reason_code": "empty_audio"}
            frame_bytes = converter.output_frame_bytes
            pcm = pcm[: len(pcm) - (len(pcm) % frame_bytes)]
            if not pcm:
                self._set_state("unavailable", "没有采集到可识别的语音")
                return {"status": "unavailable", "reason_code": "empty_audio"}
            captured = CapturedPcm(pcm, converter.sample_rate, converter.channels, reason)
            self._set_state("captured", "录音完成，准备识别")
            self.captureReady.emit(captured)
            self._apply_pending_configuration()
            return {"status": "captured", "bytes": len(pcm)}

        def abort(self, message: str, *, reason_code: str = "cancelled") -> None:
            """丢弃当前音频；错误正文和 PCM 都不会进入日志或公开状态。"""

            self._permission_generation += 1
            stop_timer = getattr(self._timer, "stop", None)
            if callable(stop_timer):
                stop_timer()
            source = self._source
            self._stopping = True
            if source is not None:
                try:
                    source.stop()
                except (AttributeError, RuntimeError):
                    pass
            self._source = None
            self._io_device = None
            self._converter = None
            self._active_device_id = ""
            self._pcm.clear()
            self._stopping = False
            self._set_state("unavailable", str(message))
            self._apply_pending_configuration()

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            self.abort("语音输入已关闭", reason_code="closed")
            if self._device_listener_connected:
                changed = getattr(self._media_devices, "audioInputsChanged", None)
                disconnect = getattr(changed, "disconnect", None)
                if callable(disconnect):
                    try:
                        disconnect(self._audio_inputs_changed)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
                self._device_listener_connected = False
            self._set_state("closed", "语音输入已关闭")

    class QtPushToTalkController(QObject):
        """协调物理播放、录音和 RuntimeLoop ASR Future 的半双工状态。"""

        stateChanged = Signal(str, str)
        transcriptionReady = Signal(str)

        def __init__(
            self,
            capture: QtMicrophoneCapture,
            *,
            transcription_submitter: Callable[[CapturedPcm], object],
            audio_player: object,
            poll_interval_ms: int = 40,
            parent: QObject | None = None,
        ) -> None:
            super().__init__(parent)
            if not callable(transcription_submitter):
                raise TypeError("microphone transcription submitter is required")
            self.capture = capture
            self._submitter = transcription_submitter
            self._audio_player = audio_player
            self._future: object | None = None
            self._pending_start = False
            self._closed = False
            self._state = capture.state
            self._message = "语音输入待命"
            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(max(10, min(int(poll_interval_ms), 1_000)))
            self._poll_timer.timeout.connect(self._poll_transcription)
            capture.stateChanged.connect(self._capture_state_changed)
            capture.captureReady.connect(self._submit_transcription)
            playback_signal = getattr(audio_player, "playbackStateChanged", None)
            connect = getattr(playback_signal, "connect", None)
            if callable(connect):
                connect(self._playback_state_changed)

        @property
        def state(self) -> str:
            return self._state

        @property
        def blocks_playback(self) -> bool:
            return self._state in {
                "waiting_playback",
                "permission_pending",
                "recording",
                "captured",
                "transcribing",
            }

        @property
        def transcribing(self) -> bool:
            return self._state == "transcribing"

        def _set_state(self, state: str, message: str) -> None:
            self._state = str(state)
            self._message = str(message)
            self.stateChanged.emit(self._state, self._message)

        def _capture_state_changed(self, state: str, message: str) -> None:
            if self._closed or self.transcribing:
                return
            self._set_state(state, message)

        def toggle(self) -> Mapping[str, object]:
            """只由用户按钮调用：播放中先停止并等待真实播放终态。"""

            if self._closed:
                return {"status": "closed"}
            if self.capture.recording:
                return self.capture.stop()
            if self._state in {"waiting_playback", "permission_pending", "transcribing"}:
                return {"status": self._state}
            # 即使当前快照显示未播放，也先设置半双工门禁并提交一次 stop_all。
            # 这样早于点击、但尚未送达 Qt 主线程的 audioReceived 排队信号会
            # 在下一事件循环检查前被清空，不会在录音开始后突然出声。
            stopper = getattr(self._audio_player, "stop_all", None)
            if not callable(stopper):
                self._set_state("unavailable", "无法停止当前语音播放")
                return {"status": "unavailable", "reason_code": "playback_stop_unavailable"}
            self._pending_start = True
            self._set_state("waiting_playback", "正在停止语音播放…")
            stopper()
            QTimer.singleShot(0, self._start_after_playback)
            return {"status": "waiting_playback"}

        def _playback_state_changed(self, playing: bool) -> None:
            if not bool(playing) and self._pending_start and not self._closed:
                QTimer.singleShot(0, self._start_after_playback)

        def _start_after_playback(self) -> None:
            if self._closed or not self._pending_start:
                return
            if bool(getattr(self._audio_player, "playing", False)):
                stopper = getattr(self._audio_player, "stop_all", None)
                if callable(stopper):
                    stopper()
                return
            self._pending_start = False
            self.capture.request_start()

        def _submit_transcription(self, captured: CapturedPcm) -> None:
            if self._closed:
                return
            self._set_state("transcribing", "正在识别语音…")
            try:
                future = self._submitter(captured)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                self._set_transcription_failure("语音识别未能启动")
                return
            self._future = future
            done = getattr(future, "done", None)
            if callable(done):
                self._poll_timer.start()
                self._poll_transcription()
                return
            self._finish_transcription(future)

        def _poll_transcription(self) -> None:
            future = self._future
            if future is None:
                self._poll_timer.stop()
                return
            done = getattr(future, "done", None)
            if not callable(done) or not done():
                return
            self._future = None
            self._poll_timer.stop()
            try:
                result = future.result()
            except BaseException:
                if not self._closed:
                    self._set_transcription_failure("语音识别失败，请重试")
                return
            self._finish_transcription(result)

        def _set_transcription_failure(self, message: str) -> None:
            if self.capture.public_status().get("enabled") is False:
                self._set_state("disabled", "语音输入已关闭")
            else:
                self._set_state("unavailable", message)

        def _finish_transcription(self, result: object) -> None:
            if self._closed:
                return
            text = str(getattr(result, "text", "") or "").strip()
            if not text and isinstance(result, Mapping):
                text = str(result.get("text", "") or "").strip()
            if not text:
                self._set_transcription_failure("没有识别到可填写的文字")
                return
            # 对话服务的明确输入上限为 20_000 字；不自动提交，正文只通过
            # 进程内 Qt 信号填入当前控制台输入框。
            text = text[:20_000]
            self.transcriptionReady.emit(text)
            capture_status = self.capture.public_status()
            if capture_status.get("enabled") is False:
                self._set_state("disabled", "识别完成，语音输入已关闭")
            else:
                self._set_state("completed", "识别完成，文字已填入输入框")

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            self._pending_start = False
            self._poll_timer.stop()
            future = self._future
            self._future = None
            cancel = getattr(future, "cancel", None)
            try:
                if callable(cancel):
                    cancel()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                # Future 属于外部 RuntimeLoop；即使其取消接口异常，也必须继续
                # 关闭采集器并发布闭合终态，避免退出路径卡在 transcribing。
                pass
            finally:
                self.capture.close()
                self._set_state("closed", "语音输入已关闭")


elif _QT_CORE_AVAILABLE:

    class QtMicrophoneCapture(QObject):  # pragma: no cover
        """Qt Core 可用但 Multimedia 缺失时的明确降级实现。"""

        stateChanged = Signal(str, str)
        captureReady = Signal(object)
        inputDevicesChanged = Signal(object)

        def __init__(
            self, *_args: object, parent: QObject | None = None, **_kwargs: object
        ) -> None:
            super().__init__(parent)
            self.state = "unavailable"
            self.recording = False
            self.busy = False
            self.input_devices_loaded = False
            self.input_device_choices: tuple[AudioInputDeviceChoice, ...] = ()

        def request_start(self) -> Mapping[str, object]:
            self.stateChanged.emit("unavailable", "Qt Multimedia 不可用")
            return {"status": "unavailable", "reason_code": "qt_multimedia_unavailable"}

        def stop(self, *, reason: str = "user") -> Mapping[str, object]:
            del reason
            return {"status": "unavailable"}

        def configure(self, **_values: object) -> None:
            return None

        def public_status(self) -> Mapping[str, object]:
            return {"state": "unavailable", "enabled": False, "recording": False}

        def load_input_devices(self) -> tuple[AudioInputDeviceChoice, ...]:
            return ()

        def close(self) -> None:
            self.state = "closed"
            self.stateChanged.emit("closed", "语音输入已关闭")

    class QtPushToTalkController(QObject):  # pragma: no cover
        """Qt Multimedia 缺失时保持控制台可恢复的降级控制器。"""

        stateChanged = Signal(str, str)
        transcriptionReady = Signal(str)

        def __init__(
            self,
            capture: QtMicrophoneCapture,
            *,
            transcription_submitter: Callable[[CapturedPcm], object],
            audio_player: object,
            parent: QObject | None = None,
            **_kwargs: object,
        ) -> None:
            super().__init__(parent)
            del transcription_submitter, audio_player
            self.capture = capture
            self.state = "unavailable"
            self.blocks_playback = False
            self.transcribing = False

        def toggle(self) -> Mapping[str, object]:
            self.stateChanged.emit("unavailable", "Qt Multimedia 不可用")
            return {"status": "unavailable", "reason_code": "qt_multimedia_unavailable"}

        def close(self) -> None:
            self.capture.close()
            self.state = "closed"
            self.stateChanged.emit("closed", "语音输入已关闭")


else:

    class QtMicrophoneCapture:  # pragma: no cover
        """无 Qt Multimedia 时的明确不可用实现。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("Qt microphone capture is unavailable")

    class QtPushToTalkController:  # pragma: no cover
        """无 Qt Multimedia 时的明确不可用实现。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("Qt microphone capture is unavailable")


__all__ = [
    "AudioInputDeviceChoice",
    "CapturedPcm",
    "QtMicrophoneCapture",
    "QtPushToTalkController",
]
