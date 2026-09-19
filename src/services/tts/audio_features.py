from __future__ import annotations

import math
import struct
import threading
from collections.abc import Iterable

from core.events.types import AudioChunk, AudioFeature, ConversationContext


class PcmAudioFeatureAnalyzer:
    source = "pcm-rms"

    def __init__(
        self,
        *,
        silence_threshold: float = 0.02,
        open_threshold: float = 0.35,
    ) -> None:
        silence = float(silence_threshold)
        opened = float(open_threshold)
        if not math.isfinite(silence) or not math.isfinite(opened):
            raise ValueError("audio thresholds must be finite")
        if not 0.0 <= silence < opened <= 1.0:
            raise ValueError("audio thresholds must satisfy 0 <= silence < open <= 1")
        self.silence_threshold = silence
        self.open_threshold = opened
        # 每个请求单独累计 PCM 帧的时间偏移，确保连续音频块的口型事件
        # 能按真实播放时间顺序排列；锁用于跨事件循环的并发分析调用。
        self._offsets: dict[tuple[ConversationContext, str], tuple[int, int]] = {}
        self._offset_lock = threading.RLock()

    def analyze(self, chunk: AudioChunk) -> Iterable[AudioFeature] | None:
        """返回当前块的口型特征；格式未知、空块或未对齐块返回 ``None``。"""

        if not self._valid_chunk(chunk):
            return None
        data = chunk.data
        channels = chunk.channels
        frame_width = channels * 2
        key = (chunk.context, chunk.request_id)
        if not data or len(data) % frame_width:
            if chunk.is_final:
                with self._offset_lock:
                    self._offsets.pop(key, None)
            return None
        frame_count = len(data) // frame_width
        if frame_count <= 0:
            if chunk.is_final:
                with self._offset_lock:
                    self._offsets.pop(key, None)
            return None

        try:
            samples = struct.iter_unpack("<" + ("h" * channels), data)
            square_sum = 0
            for frame in samples:
                if channels == 1:
                    value = frame[0]
                else:
                    value = (frame[0] + frame[1]) / 2.0
                square_sum += value * value
        except (TypeError, ValueError, struct.error, OverflowError):
            if chunk.is_final:
                with self._offset_lock:
                    self._offsets.pop(key, None)
            return None

        rms = math.sqrt(square_sum / frame_count) / 32768.0
        if not math.isfinite(rms):
            if chunk.is_final:
                with self._offset_lock:
                    self._offsets.pop(key, None)
            return None
        intensity = max(0.0, min(1.0, rms))
        if rms <= self.silence_threshold:
            viseme = "Silence"
            intensity = 0.0
        elif rms >= self.open_threshold:
            viseme = "Open"
        else:
            viseme = "Neutral"

        # 用帧数而非逐块四舍五入后的毫秒数累计，避免长音频产生时间漂移。
        with self._offset_lock:
            previous_frames, previous_rate = self._offsets.get(key, (0, chunk.sample_rate))
            if previous_rate == chunk.sample_rate:
                start_ms = int(round(previous_frames * 1000 / chunk.sample_rate))
            else:
                elapsed_ms = int(round(previous_frames * 1000 / previous_rate))
                start_ms = elapsed_ms
                previous_frames = int(round(elapsed_ms * chunk.sample_rate / 1000))
            self._offsets[key] = (previous_frames + frame_count, chunk.sample_rate)
            if chunk.is_final:
                self._offsets.pop(key, None)

        duration_ms = max(1, int(round(frame_count * 1000 / chunk.sample_rate)))
        try:
            return (
                AudioFeature(
                    context=chunk.context,
                    request_id=chunk.request_id,
                    viseme=viseme,
                    intensity=intensity,
                    start_ms=start_ms,
                    duration_ms=duration_ms,
                    source=self.source,
                ),
            )
        except (TypeError, ValueError, OverflowError):
            return None

    def reset(self, context: ConversationContext, request_id: str) -> None:
        """释放未收到终态的请求状态，例如后端失败或取消后的流。"""

        if not isinstance(context, ConversationContext):
            return
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            return
        with self._offset_lock:
            self._offsets.pop((context, normalized_request_id), None)

    @staticmethod
    def _valid_chunk(chunk: object) -> bool:
        """严格验证分析器真正消费的事件字段。"""

        if not isinstance(chunk, AudioChunk):
            return False
        if not isinstance(chunk.context, ConversationContext):
            return False
        if not isinstance(chunk.request_id, str) or not chunk.request_id.strip():
            return False
        if chunk.sample_format != "s16le":
            return False
        if not isinstance(chunk.sample_rate, int) or isinstance(chunk.sample_rate, bool):
            return False
        if chunk.sample_rate <= 0:
            return False
        if isinstance(chunk.channels, bool) or chunk.channels not in (1, 2):
            return False
        return isinstance(chunk.data, bytes)


__all__ = ["PcmAudioFeatureAnalyzer"]
