from __future__ import annotations

import struct
from dataclasses import dataclass


class AudioProtocolError(ValueError):
    """上游音频不符合受支持的 PCM/WAV 协议。"""


@dataclass(frozen=True)
class PcmFormat:
    """播放器支持的线性 PCM 格式。"""

    sample_rate: int
    channels: int
    sample_format: str = "s16le"

    def __post_init__(self) -> None:
        sample_rate = int(self.sample_rate)
        channels = int(self.channels)
        sample_format = str(self.sample_format or "").strip().lower()
        if not 1 <= sample_rate <= 384000:
            raise AudioProtocolError("PCM sample rate is outside the supported range")
        if channels not in {1, 2}:
            raise AudioProtocolError("PCM channel count must be 1 or 2")
        if sample_format != "s16le":
            raise AudioProtocolError("only s16le PCM is supported")
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "sample_format", sample_format)

    @property
    def frame_bytes(self) -> int:
        return self.channels * 2


class StreamingPcmValidator:
    """校验任意分片边界下的 S16LE PCM，并按完整音频帧发出数据。"""

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        *,
        max_total_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.format = PcmFormat(sample_rate, channels)
        self.max_total_bytes = max(1, int(max_total_bytes))
        self._remainder = bytearray()
        self._total_bytes = 0
        self._emitted_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def feed(self, data: bytes | bytearray | memoryview) -> tuple[bytes, ...]:
        value = bytes(data)
        if not value:
            return ()
        self._total_bytes += len(value)
        if self._total_bytes > self.max_total_bytes:
            raise AudioProtocolError("PCM stream exceeds the configured size limit")
        if self._remainder:
            value = bytes(self._remainder) + value
            self._remainder.clear()
        frame_bytes = self.format.frame_bytes
        complete = len(value) - (len(value) % frame_bytes)
        if complete < len(value):
            self._remainder.extend(value[complete:])
        if complete == 0:
            return ()
        payload = value[:complete]
        self._emitted_bytes += len(payload)
        return (payload,)

    def finish(self, *, require_audio: bool = True) -> PcmFormat:
        if self._remainder:
            raise AudioProtocolError("PCM stream ends in a partial audio frame")
        if require_audio and self._emitted_bytes == 0:
            raise AudioProtocolError("PCM stream contains no audio data")
        return self.format


class StreamingWavDecoder:
    """增量解析 RIFF/WAVE PCM 流。

    支持 ``JUNK``、``LIST`` 等未知 RIFF 块，并允许 ``data`` 块长度为零或
    ``0xffffffff``，这两种写法常见于无法预先知道总长度的实时服务。遇到
    ``data`` 块后，未知长度数据被视为流尾，避免把后续裸 PCM 错当成块头。
    ``expected_sample_rate`` 非空时，在第一个音频帧发出前锁定采样率，防止
    服务端切换模型或配置后把不兼容的音频送入播放器。
    """

    def __init__(
        self,
        *,
        max_total_bytes: int = 64 * 1024 * 1024,
        max_format_bytes: int = 1024 * 1024,
        expected_sample_rate: int | None = None,
    ) -> None:
        self.max_total_bytes = max(1, int(max_total_bytes))
        self.max_format_bytes = max(16, int(max_format_bytes))
        if expected_sample_rate is not None:
            if isinstance(expected_sample_rate, bool):
                raise AudioProtocolError("expected WAV sample rate must be an integer")
            try:
                normalized_sample_rate = int(expected_sample_rate)
            except (TypeError, ValueError, OverflowError) as exc:
                raise AudioProtocolError("expected WAV sample rate must be an integer") from exc
            if normalized_sample_rate != expected_sample_rate:
                raise AudioProtocolError("expected WAV sample rate must be an integer")
            if not 1 <= normalized_sample_rate <= 384000:
                raise AudioProtocolError("expected WAV sample rate is outside the supported range")
            self.expected_sample_rate: int | None = normalized_sample_rate
        else:
            self.expected_sample_rate = None
        self._buffer = bytearray()
        self._riff_seen = False
        self._chunk_id: bytes | None = None
        self._chunk_remaining = 0
        self._chunk_padding = 0
        self._pending_padding = 0
        self._fmt_data = bytearray()
        self._format: PcmFormat | None = None
        self._pcm: StreamingPcmValidator | None = None
        self._saw_data = False
        self._data_open_ended = False
        self._finished = False
        self._total_input = 0

    @property
    def format(self) -> PcmFormat | None:
        return self._format

    @property
    def saw_audio(self) -> bool:
        return bool(self._pcm is not None and self._pcm.total_bytes > 0)

    def feed(self, data: bytes | bytearray | memoryview) -> tuple[bytes, ...]:
        if self._finished:
            raise AudioProtocolError("WAV decoder has already finished")
        value = bytes(data)
        if not value:
            return ()
        self._total_input += len(value)
        if self._total_input > self.max_total_bytes + 12:
            raise AudioProtocolError("WAV stream exceeds the configured size limit")
        self._buffer.extend(value)
        emitted: list[bytes] = []
        while True:
            if not self._riff_seen:
                if len(self._buffer) < 12:
                    break
                if self._buffer[:4] != b"RIFF" or self._buffer[8:12] != b"WAVE":
                    raise AudioProtocolError("WAV stream has an invalid RIFF/WAVE header")
                # RIFF size is advisory for a live stream; validate its presence but
                # do not require it to equal the eventual byte count.
                riff_size = struct.unpack_from("<I", self._buffer, 4)[0]
                if riff_size not in {0, 0xFFFFFFFF} and riff_size < 4:
                    raise AudioProtocolError("WAV RIFF size is invalid")
                del self._buffer[:12]
                self._riff_seen = True

            if self._data_open_ended:
                emitted.extend(self._feed_pcm(bytes(self._buffer)))
                self._buffer.clear()
                break

            if self._chunk_remaining:
                if not self._buffer:
                    break
                take = min(len(self._buffer), self._chunk_remaining)
                part = bytes(self._buffer[:take])
                del self._buffer[:take]
                self._chunk_remaining -= take
                if self._chunk_id == b"data":
                    emitted.extend(self._feed_pcm(part))
                elif self._chunk_id == b"fmt ":
                    if len(self._fmt_data) + len(part) > self.max_format_bytes:
                        raise AudioProtocolError("WAV fmt chunk is too large")
                    self._fmt_data.extend(part)
                if self._chunk_remaining:
                    break
                if self._chunk_id == b"fmt ":
                    self._parse_format()
                self._chunk_id = None
                self._chunk_padding = 1 if (self._chunk_remaining & 1) else 0
                # ``_chunk_remaining`` is zero here; use a separate value because
                # the original size is needed to determine odd-byte padding.
                self._chunk_padding = self._pending_padding
                continue

            if self._chunk_padding:
                if not self._buffer:
                    break
                del self._buffer[:1]
                self._chunk_padding = 0
                continue

            if len(self._buffer) < 8:
                break
            chunk_id = bytes(self._buffer[:4])
            chunk_size = struct.unpack_from("<I", self._buffer, 4)[0]
            del self._buffer[:8]
            if chunk_id == b"data":
                if self._format is None:
                    raise AudioProtocolError("WAV data chunk appears before fmt chunk")
                self._saw_data = True
                self._chunk_id = chunk_id
                if chunk_size in {0, 0xFFFFFFFF}:
                    self._data_open_ended = True
                    if self._buffer:
                        emitted.extend(self._feed_pcm(bytes(self._buffer)))
                        self._buffer.clear()
                    break
            self._chunk_id = chunk_id
            self._chunk_remaining = chunk_size
            self._pending_padding = 1 if (chunk_size & 1) else 0
            if chunk_size == 0:
                if chunk_id == b"fmt ":
                    self._parse_format()
                self._chunk_id = None
                self._chunk_padding = self._pending_padding
        return tuple(emitted)

    def _feed_pcm(self, data: bytes) -> tuple[bytes, ...]:
        if self._pcm is None:
            raise AudioProtocolError("WAV PCM format is not available")
        return self._pcm.feed(data)

    def _parse_format(self) -> None:
        if self._format is not None:
            return
        if len(self._fmt_data) < 16:
            raise AudioProtocolError("WAV fmt chunk is truncated")
        audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from(
            "<HHIIHH", self._fmt_data
        )
        if audio_format != 1:
            raise AudioProtocolError("WAV format is not linear PCM")
        if bits != 16:
            raise AudioProtocolError("WAV PCM must use 16-bit samples")
        expected_align = channels * (bits // 8)
        if block_align != expected_align or byte_rate != sample_rate * block_align:
            raise AudioProtocolError("WAV PCM byte rate or block alignment is invalid")
        self._format = PcmFormat(sample_rate, channels)
        if (
            self.expected_sample_rate is not None
            and self._format.sample_rate != self.expected_sample_rate
        ):
            raise AudioProtocolError(
                f"WAV sample rate {self._format.sample_rate} does not match "
                f"expected {self.expected_sample_rate}"
            )
        self._pcm = StreamingPcmValidator(
            sample_rate,
            channels,
            max_total_bytes=self.max_total_bytes,
        )

    def finish(self) -> PcmFormat:
        if self._finished:
            if self._format is None:
                raise AudioProtocolError("WAV stream has no PCM format")
            return self._format
        if not self._riff_seen:
            raise AudioProtocolError("WAV stream is empty or truncated")
        if self._buffer and not self._data_open_ended:
            raise AudioProtocolError("WAV stream ends in a truncated RIFF chunk")
        if self._chunk_remaining and not self._data_open_ended:
            raise AudioProtocolError("WAV stream ends before a RIFF chunk completed")
        if self._format is None:
            raise AudioProtocolError("WAV stream has no PCM fmt chunk")
        if not self._saw_data or self._pcm is None:
            raise AudioProtocolError("WAV stream has no PCM data chunk")
        # 先校验 PCM 帧完整性，让数据块本身的奇数长度继续报告更直接的
        # partial audio frame；随后再检查元数据块的 RIFF 对齐字节。
        audio_format = self._pcm.finish()
        if self._chunk_padding and not self._data_open_ended:
            # RIFF 奇数长度块必须带一个对齐字节；若流恰好在该字节前结束，
            # 后续块边界无法恢复，不能把这份音频报告为完整。
            raise AudioProtocolError("WAV stream ends before RIFF chunk padding")
        # 只有全部校验通过后才标记完成。若校验失败，下一次 finish() 必须
        # 继续报告同一协议错误，不能因为半成品状态被标记为 finished 而误报成功。
        self._finished = True
        return audio_format


__all__ = [
    "AudioProtocolError",
    "PcmFormat",
    "StreamingPcmValidator",
    "StreamingWavDecoder",
]
