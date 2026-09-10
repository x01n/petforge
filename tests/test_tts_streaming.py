from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from core.events.types import AudioChunk, ConversationContext
from core.tts.contracts import EngineHealth, SpeechChunk, SpeechRequest
from gui.qt6.audio import QtAudioPlayer
from logger.events import event_fingerprint
from services.tts.backend import SubprocessTTSBackend, TextOnlyBackend
from services.tts.coordinator import SpeechSegmenter, TTSCoordinator


def test_segmenter_flushes_sentences_and_long_text() -> None:
    segmenter = SpeechSegmenter(max_chars=20)
    assert segmenter.push("你好。") == ("你好。",)
    assert segmenter.push("abcdefghijklmnopqrstuvw") == ("abcdefghijklmnopqrst",)
    assert segmenter.flush() == ("uvw",)


def test_segmenter_hard_limit_applies_when_sentence_boundary_is_late() -> None:
    segmenter = SpeechSegmenter(max_chars=20)
    source = "a" * 19 + "，" + "b" * 20 + "。"

    segments = segmenter.push(source) + segmenter.flush()

    assert segments
    assert all(0 < len(segment) <= 20 for segment in segments)
    assert "".join(segments) == source


def test_segmenter_flushes_paragraphs_and_absorbs_sentence_closing_marks() -> None:
    segmenter = SpeechSegmenter(max_chars=20)

    assert segmenter.push("第一段\n第二段。") == ("第一段", "第二段。")
    assert segmenter.push('这是一个句子。"') == ('这是一个句子。"',)
    assert segmenter.push("省略号……下一句") == ("省略号……",)


def test_segmenter_preserves_whitespace_at_soft_boundaries() -> None:
    source = "hello world this is a test and more words"
    segmenter = SpeechSegmenter(max_chars=20)

    segments = segmenter.push(source) + segmenter.flush()

    assert "".join(segments) == source
    assert all(0 < len(segment) <= 20 for segment in segments)


def test_segmenter_keeps_decimal_urls_and_email_addresses_intact() -> None:
    segmenter = SpeechSegmenter(max_chars=120)
    source = "版本 3.14 已发布。访问 https://example.com/path。Email a.b@example.com。"

    assert segmenter.push(source) == (
        "版本 3.14 已发布。",
        "访问 https://example.com/path。",
        "Email a.b@example.com。",
    )
    assert SpeechSegmenter(max_chars=120).push("中文.下一句") == ("中文.",)


def test_segmenter_defers_trailing_period_across_model_deltas() -> None:
    segmenter = SpeechSegmenter(max_chars=120)

    assert segmenter.push("版本 3.") == ()
    assert segmenter.push("14 已发布。") == ("版本 3.14 已发布。",)

    ellipsis = SpeechSegmenter(max_chars=120)
    assert ellipsis.push("Wait.") == ()
    assert ellipsis.push(".") == ()
    assert ellipsis.push(". Next sentence.") == ("Wait...",)
    assert ellipsis.flush() == (" Next sentence.",)


def test_tts_coordinator_applies_configured_segment_limit() -> None:
    requests: list[str] = []

    class Backend:
        async def stream(self, request):
            requests.append(request.text)
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    coordinator = TTSCoordinator(Backend(), segment_max_chars=20)
    context = ConversationContext("p", "s", "configured-limit", 1)
    source = "a" * 45

    asyncio.run(coordinator.enqueue_text(context, source, flush=True))

    assert requests
    assert all(0 < len(text) <= 20 for text in requests)
    assert "".join(requests) == source


def test_tts_queue_limit_can_hot_reload_without_dropping_audio() -> None:
    coordinator = TTSCoordinator(TextOnlyBackend(), queue_size=2)
    assert coordinator.queue_size == 2
    assert coordinator.set_queue_size(8) == 8
    assert coordinator.queue_size == 8
    with pytest.raises(ValueError, match="outside"):
        coordinator.set_queue_size(0)


def test_runtime_builds_configured_tts_segmentation(tmp_path) -> None:
    from app.runtime import _build_tts
    from config.loader import LoadedConfiguration

    coordinator = _build_tts(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {
                    "enabled": False,
                    "backend": "text_only",
                    "segmentation": {
                        "max_chars": 48,
                        "hard_boundaries": "。!",
                        "soft_boundaries": "，, ",
                    },
                }
            },
        )
    )

    assert coordinator.segment_max_chars == 48
    assert coordinator.segment_hard_boundaries == "。!"
    assert coordinator.segment_soft_boundaries == "，, "


def test_segmentation_reload_preserves_an_unfinished_turn_tail() -> None:
    requests: list[str] = []

    class Backend:
        async def stream(self, request):
            requests.append(request.text)
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    coordinator = TTSCoordinator(Backend(), segment_max_chars=40)
    context = ConversationContext("p", "s", "reload-tail", 1)

    asyncio.run(coordinator.enqueue_text(context, "未结束的句子"))
    coordinator.set_segmentation(max_chars=20)
    asyncio.run(coordinator.enqueue_text(context, "。", flush=True))

    assert requests == ["未结束的句子。"]


def test_tts_coordinator_emits_bounded_audio_chunks() -> None:
    class FakeBackend:
        async def health(self):
            return EngineHealth("fake", True)

        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1)
            yield SpeechChunk(request.request_id, b"", 24000, 1, is_final=True)

    events = []
    coordinator = TTSCoordinator(FakeBackend(), audio_sink=events.append)
    context = ConversationContext("p", "s", "t", 1)
    asyncio.run(coordinator.enqueue_text(context, "你好。", flush=True))
    assert len(events) == 2
    assert events[0].data == b"pcm"


def test_tts_same_context_serializes_concurrent_enqueue_calls() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    events: list[AudioChunk] = []

    class FakeBackend:
        async def stream(self, request):
            if request.text == "A。":
                started.set()
                await release.wait()
            yield SpeechChunk(request.request_id, request.text.encode(), 24000, 1)

    coordinator = TTSCoordinator(FakeBackend(), audio_sink=events.append)
    context = ConversationContext("p", "s", "ordered", 1)

    async def scenario() -> None:
        first = asyncio.create_task(coordinator.enqueue_text(context, "A。", flush=True))
        await started.wait()
        second = asyncio.create_task(coordinator.enqueue_text(context, "B。", flush=True))
        await asyncio.sleep(0)
        assert [event.data for event in events] == []
        release.set()
        await asyncio.gather(first, second)

    asyncio.run(scenario())
    assert [event.data for event in events] == ["A。".encode(), "B。".encode()]


def test_tts_same_context_serializes_across_event_loops() -> None:
    started = threading.Event()
    release = threading.Event()
    events: list[AudioChunk] = []
    errors: list[BaseException] = []

    class Backend:
        async def stream(self, request):
            if request.text == "A。":
                started.set()
                await asyncio.to_thread(release.wait, 2.0)
            yield SpeechChunk(request.request_id, request.text.encode(), 24000, 1)

    coordinator = TTSCoordinator(Backend(), audio_sink=events.append)
    context = ConversationContext("p", "s", "cross-loop", 1)

    def run(text: str) -> None:
        try:
            asyncio.run(coordinator.enqueue_text(context, text, flush=True))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=run, args=("A。",))
    second = threading.Thread(target=run, args=("B。",))
    first.start()
    assert started.wait(timeout=1.0)
    second.start()
    time.sleep(0.05)
    release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert [event.data for event in events] == ["A。".encode(), "B。".encode()]


def test_tts_queue_full_does_not_block_synthesis() -> None:
    class FakeBackend:
        async def stream(self, request):
            for index in range(4):
                yield SpeechChunk(request.request_id, bytes([index]), 24000, 1)

    coordinator = TTSCoordinator(FakeBackend(), queue_size=2)
    context = ConversationContext("p", "s", "t", 1)
    asyncio.run(coordinator.enqueue_text(context, "一段足够长的句子。", flush=True))
    assert coordinator._queue.qsize() == 2


def test_qt_audio_player_is_safe_without_multimedia() -> None:
    player = QtAudioPlayer()
    player.close()


def test_status_sink_failure_does_not_stop_synthesis() -> None:
    class FakeBackend:
        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    def broken_status(_status):
        raise RuntimeError("status sink unavailable")

    audio = []
    coordinator = TTSCoordinator(
        FakeBackend(),
        status_sink=broken_status,
        audio_sink=audio.append,
    )
    context = ConversationContext("p", "s", "t", 1)
    asyncio.run(coordinator.enqueue_text(context, "你好。", flush=True))
    assert audio and audio[0].data == b"pcm"


def test_slow_status_sink_does_not_delay_first_audio_chunk() -> None:
    started_in_sink = asyncio.Event()
    release_status = asyncio.Event()
    backend_started = asyncio.Event()
    audio_received = asyncio.Event()
    statuses = []

    class Backend:
        async def stream(self, request):
            backend_started.set()
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    async def status_sink(status) -> None:
        statuses.append(status.state)
        if status.state == "started":
            started_in_sink.set()
            await release_status.wait()

    def audio_sink(_chunk: AudioChunk) -> None:
        audio_received.set()

    coordinator = TTSCoordinator(
        Backend(),
        status_sink=status_sink,
        audio_sink=audio_sink,
    )
    context = ConversationContext("p", "s", "status-backpressure", 1)

    async def scenario() -> None:
        task = asyncio.create_task(coordinator.enqueue_text(context, "你好。", flush=True))
        await asyncio.wait_for(started_in_sink.wait(), timeout=1.0)
        await asyncio.wait_for(backend_started.wait(), timeout=1.0)
        await asyncio.wait_for(audio_received.wait(), timeout=1.0)
        release_status.set()
        await asyncio.wait_for(task, timeout=1.0)
        await coordinator.aclose()

    asyncio.run(scenario())
    assert statuses == ["started", "completed"]


def test_tts_backend_failure_emits_degraded_status_without_blocking_text() -> None:
    class BrokenBackend:
        async def stream(self, _request):
            raise RuntimeError("engine unavailable")
            yield  # 保持异步生成器契约

    statuses = []
    coordinator = TTSCoordinator(BrokenBackend(), status_sink=statuses.append)
    context = ConversationContext("p", "s", "degraded", 1)
    segments = asyncio.run(coordinator.enqueue_text(context, "你好。", flush=True))
    assert segments == ("你好。",)
    assert [status.state for status in statuses] == ["started", "degraded"]
    assert statuses[-1].message == "语音不可用，保留文本输出"


def test_aclose_cancels_active_speech_task() -> None:
    started = asyncio.Event()

    class BlockingBackend:
        async def stream(self, request):
            started.set()
            await asyncio.Event().wait()
            if False:
                yield SpeechChunk(request.request_id, b"", 24000, 1)

    coordinator = TTSCoordinator(BlockingBackend())
    context = ConversationContext("p", "s", "t", 1)

    async def scenario():
        task = asyncio.create_task(coordinator.enqueue_text(context, "你好。", flush=True))
        await started.wait()
        await coordinator.aclose()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("speech task completed unexpectedly")

    asyncio.run(scenario())


def test_coordinator_close_cancels_active_task_owned_by_another_event_loop() -> None:
    """从配置线程关闭时，不能直接把异步任务交给当前事件循环 await。"""

    started = threading.Event()
    backend_closed = threading.Event()
    errors: list[BaseException] = []
    context = ConversationContext("p", "s", "cross-loop-close", 1)

    class BlockingBackend:
        async def stream(self, request):
            started.set()
            await asyncio.Event().wait()
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

        async def aclose(self) -> None:
            backend_closed.set()

    coordinator = TTSCoordinator(BlockingBackend())

    def run_speech() -> None:
        async def scenario() -> None:
            task = asyncio.create_task(coordinator.enqueue_text(context, "你好。", flush=True))
            try:
                await task
            except asyncio.CancelledError:
                pass

        try:
            asyncio.run(scenario())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_speech)
    thread.start()
    assert started.wait(timeout=1.0)
    asyncio.run(coordinator.aclose())
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert errors == []
    assert backend_closed.is_set()
    assert coordinator.closed is True


def test_subprocess_tts_cancellation_preserves_cancelled_error() -> None:
    backend = SubprocessTTSBackend(
        (
            sys.executable,
            "-c",
            "import sys, time; sys.stdin.readline(); time.sleep(30)",
        ),
        require_ready=False,
    )

    async def scenario():
        stream = backend.stream(SpeechRequest("request", "你好"))
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("subprocess speech task completed unexpectedly")
        await stream.aclose()

    asyncio.run(scenario())


def test_subprocess_tts_aclose_terminates_worker_immediately() -> None:
    payload = json.dumps(
        {
            "type": "audio",
            "protocol": "meapet.tts.jsonl",
            "version": 1,
            "request_id": "request",
            "data": base64.b64encode(b"pcm").decode("ascii"),
            "sample_rate": 24000,
            "channels": 1,
        }
    )
    backend = SubprocessTTSBackend(
        (
            sys.executable,
            "-c",
            "import sys,time; print(sys.argv[1], flush=True); time.sleep(30)",
            payload,
        ),
        require_ready=False,
    )

    async def scenario():
        stream = backend.stream(SpeechRequest("request", "你好"))
        chunk = await anext(stream)
        assert chunk.data == b"pcm"
        started = time.monotonic()
        await stream.aclose()
        assert time.monotonic() - started < 1.0

    asyncio.run(scenario())


def test_subprocess_tts_cancellation_terminates_child_process_group(tmp_path) -> None:
    if os.name != "posix":
        return
    child_pid_path = tmp_path / "child.pid"
    worker = (
        "import pathlib, subprocess, sys, time; "
        "sys.stdin.readline(); "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
    )
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker, str(child_pid_path)),
        require_ready=False,
    )

    def child_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    async def scenario() -> int:
        stream = backend.stream(SpeechRequest("request", "你好"))
        task = asyncio.create_task(anext(stream))
        deadline = time.monotonic() + 2.0
        while not child_pid_path.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert child_pid_path.exists()
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await stream.aclose()
        return child_pid

    child_pid = asyncio.run(scenario())
    deadline = time.monotonic() + 2.0
    while child_alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not child_alive(child_pid)


def test_subprocess_tts_reaps_child_holding_stdout_after_parent_exit(tmp_path) -> None:
    if os.name != "posix":
        return
    child_pid_path = tmp_path / "tts-pipe-child.pid"
    worker = (
        "import pathlib, subprocess, sys; "
        "sys.stdin.readline(); "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))"
    )
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker, str(child_pid_path)),
        require_ready=False,
    )

    async def scenario() -> None:
        stream = backend.stream(SpeechRequest("request", "你好"))
        try:
            await asyncio.wait_for(anext(stream), timeout=2.0)
        except RuntimeError as exc:
            assert "stdout" in str(exc)
        finally:
            await stream.aclose()

    asyncio.run(scenario())
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("TTS child holding stdout survived parent exit")


def test_audio_observer_queue_crosses_event_loop_threads() -> None:
    class FakeBackend:
        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    coordinator = TTSCoordinator(FakeBackend())
    context = ConversationContext("p", "s", "t", 1)
    ready = threading.Event()
    waiting = threading.Event()

    original_next_audio = coordinator.next_audio

    async def observed_next_audio():
        waiting.set()
        return await original_next_audio()

    coordinator.next_audio = observed_next_audio  # type: ignore[method-assign]
    received: list[bytes] = []
    failure: list[BaseException] = []

    def consume() -> None:
        async def scenario() -> None:
            ready.set()
            chunk = await asyncio.wait_for(coordinator.next_audio(), timeout=1.0)
            received.append(chunk.data)

        try:
            asyncio.run(scenario())
        except BaseException as exc:
            failure.append(exc)

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert ready.wait(timeout=1.0)
    assert waiting.wait(timeout=1.0)
    time.sleep(0.05)
    asyncio.run(coordinator.enqueue_text(context, "你好。", flush=True))
    consumer.join(timeout=1.0)
    assert not consumer.is_alive()
    assert not failure
    assert received == [b"pcm"]


def test_next_audio_does_not_clear_notification_for_a_newly_queued_chunk() -> None:
    """消费者唤醒与生产者入队的交错不能吞掉音频通知。"""

    coordinator = TTSCoordinator(TextOnlyBackend())
    context = ConversationContext("p", "s", "notification-race", 1)
    first_wait = threading.Event()
    release_wait = threading.Event()
    original_wait = coordinator._audio_available.wait

    def delayed_wait(timeout: float | None = None) -> bool:
        first_wait.set()
        release_wait.wait(timeout=1.0)
        return original_wait(timeout)

    coordinator._audio_available.wait = delayed_wait  # type: ignore[method-assign]

    async def scenario() -> AudioChunk:
        task = asyncio.create_task(coordinator.next_audio())
        await asyncio.to_thread(first_wait.wait, 1.0)
        coordinator._put_audio(AudioChunk(context, b"new", 24000, 1))
        release_wait.set()
        return await asyncio.wait_for(task, timeout=1.0)

    chunk = asyncio.run(scenario())
    assert chunk.data == b"new"
    assert coordinator._audio_available.is_set()


def test_cancel_drops_queued_audio_for_the_cancelled_context() -> None:
    class Sink:
        def __init__(self):
            self.cancelled = []
            self.cleared = []

        def cancel(self, context):
            self.cancelled.append(context)

        def clear_cancelled(self, context):
            self.cleared.append(context)

    sink = Sink()
    coordinator = TTSCoordinator(TextOnlyBackend(), audio_sink=sink)
    context = ConversationContext("p", "s", "t", 1)
    coordinator._put_audio(AudioChunk(context, b"old", 24000, 1))
    coordinator.cancel(context)
    assert coordinator._queue.empty()
    assert sink.cancelled == [context]
    coordinator.clear_cancelled(context)
    assert sink.cleared == [context]


def test_tts_context_key_isolated_by_mode_and_generation() -> None:
    coordinator = TTSCoordinator(TextOnlyBackend())
    direct = ConversationContext("p", "s", "same-turn", 1, "direct")
    agent = ConversationContext("p", "s", "same-turn", 1, "agent")

    async def scenario() -> None:
        await coordinator.enqueue_text(direct, "未结束")
        await coordinator.enqueue_text(agent, "另一段")

    asyncio.run(scenario())
    assert len(coordinator._segments) == 2
    coordinator.cancel(direct)
    assert len(coordinator._segments) == 1
    assert next(iter(coordinator._segments)) == coordinator._context_key(agent)


def test_tts_cancel_race_removes_audio_inserted_before_cancellation() -> None:
    class FakeBackend:
        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    coordinator = TTSCoordinator(FakeBackend())
    context = ConversationContext("p", "s", "race", 1)
    entered = threading.Event()
    release = threading.Event()
    original_put = coordinator._put_audio

    def blocked_put(event: AudioChunk) -> None:
        entered.set()
        release.wait(timeout=1)
        original_put(event)

    coordinator._put_audio = blocked_put  # type: ignore[method-assign]

    async def scenario() -> None:
        task = asyncio.create_task(
            coordinator._synthesize(context, "你好", language="zh", mood="neutral")
        )
        await asyncio.sleep(0)
        assert entered.wait(timeout=1)
        cancel_thread = threading.Thread(target=coordinator.cancel, args=(context,))
        cancel_thread.start()
        await asyncio.sleep(0.02)
        release.set()
        cancel_thread.join(timeout=1)
        await task

    asyncio.run(scenario())
    assert coordinator._queue.empty()


def test_tts_cancel_after_queue_insert_does_not_call_audio_sink() -> None:
    class FakeBackend:
        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    audio: list[AudioChunk] = []
    coordinator = TTSCoordinator(FakeBackend(), audio_sink=audio.append)
    context = ConversationContext("p", "s", "sink-race", 1)
    original_put = coordinator._put_audio

    def cancel_after_insert(event: AudioChunk) -> None:
        original_put(event)
        coordinator.cancel(context)

    coordinator._put_audio = cancel_after_insert  # type: ignore[method-assign]

    asyncio.run(coordinator._synthesize(context, "你好", language="zh", mood="neutral"))

    assert audio == []
    assert coordinator._queue.empty()


def test_tts_cancel_does_not_wait_for_a_blocking_sync_sink() -> None:
    entered = threading.Event()
    release = threading.Event()

    class FakeBackend:
        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    def blocking_sink(_event: AudioChunk) -> None:
        entered.set()
        release.wait(timeout=5.0)

    coordinator = TTSCoordinator(FakeBackend(), audio_sink=blocking_sink)
    context = ConversationContext("p", "s", "blocking-sink-cancel", 1)

    async def scenario() -> None:
        task = asyncio.create_task(coordinator.enqueue_text(context, "你好。", flush=True))
        assert await asyncio.to_thread(entered.wait, 1.0)
        coordinator.cancel(context)
        await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_sync_sink_failure_during_cancel_keeps_cancelled_terminal_state() -> None:
    entered = threading.Event()
    release = threading.Event()

    def failing_sink(_value: object) -> None:
        entered.set()
        release.wait(timeout=1.0)
        raise RuntimeError("late sink failure")

    async def scenario() -> None:
        task = asyncio.create_task(TTSCoordinator._invoke_sync(failing_sink, object()))
        assert await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_next_audio_skips_cancelled_chunk_after_queue_handoff_boundary() -> None:
    coordinator = TTSCoordinator(TextOnlyBackend())
    cancelled = ConversationContext("p", "s", "cancelled", 1)
    current = ConversationContext("p", "s", "current", 1)
    coordinator.cancel(cancelled)
    coordinator._queue.put(AudioChunk(cancelled, b"old", 24000, 1))
    coordinator._queue.put(AudioChunk(current, b"new", 24000, 1))

    chunk = asyncio.run(coordinator.next_audio())
    assert chunk.context == current
    assert chunk.data == b"new"


def test_tts_closes_streams_with_sync_aclose_contract() -> None:
    class Stream:
        def __init__(self):
            self.closed = False
            self.sent = False
            self.request_id = ""

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return SpeechChunk(self.request_id, b"pcm", 24000, 1, is_final=True)

        def aclose(self):
            self.closed = True

    stream = Stream()

    class Backend:
        def stream(self, request):
            stream.request_id = request.request_id
            return stream

    coordinator = TTSCoordinator(Backend())
    context = ConversationContext("p", "s", "sync-close", 1)
    asyncio.run(coordinator.enqueue_text(context, "你好。", flush=True))
    assert stream.closed


def test_subprocess_worker_receives_language_reference_and_model_options(tmp_path) -> None:
    ref_dir = tmp_path / "refs"
    mood_dir = ref_dir / "soft"
    mood_dir.mkdir(parents=True)
    ref_path = mood_dir / "jp_soft.wav"
    ref_path.write_bytes(b"reference")
    ref_path.with_suffix(".txt").write_text("こんにちは", encoding="utf-8")
    worker = (
        "import base64,json,sys; "
        "print(json.dumps({'type':'ready','protocol':'meapet.tts.jsonl',"
        "'version':1,'streaming':True}),flush=True); "
        "p=json.loads(sys.stdin.readline()); "
        "assert p['language']=='ja'; "
        "assert p['reference_audio'].endswith('jp_soft.wav'); "
        "assert p['reference_text']=='こんにちは'; "
        "assert p['options']['text_language']=='ja'; "
        "assert p['options']['prompt_language']=='ja'; "
        "assert p['options']['gpt_path']=='/models/gpt.ckpt'; "
        "assert p['options']['sovits_path']=='/models/sovits.pth'; "
        "print(json.dumps({'type':'audio','protocol':'meapet.tts.jsonl','version':1,"
        "'request_id':p['request_id'],"
        "'data':base64.b64encode(b'\\x00\\x00').decode(),"
        "'sample_rate':24000,'channels':1,'final':True}),flush=True)"
    )
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker),
        ref_dir=ref_dir,
        gpt_path="/models/gpt.ckpt",
        sovits_path="/models/sovits.pth",
    )

    async def scenario() -> list[SpeechChunk]:
        return [
            chunk
            async for chunk in backend.stream(
                SpeechRequest("language-model", "こんにちは", language="ja", mood="soft")
            )
        ]

    chunks = asyncio.run(scenario())
    assert chunks[-1].is_final
    assert chunks[0].data == b"\x00\x00"


def test_subprocess_worker_timeout_is_reported_and_reaped() -> None:
    backend = SubprocessTTSBackend(
        (
            sys.executable,
            "-c",
            "import json,sys,time; "
            "print(json.dumps({'type':'ready','protocol':'meapet.tts.jsonl',"
            "'version':1,'streaming':True}),flush=True); "
            "sys.stdin.readline(); time.sleep(2)",
        ),
        timeout_seconds=1,
    )

    async def scenario() -> None:
        stream = backend.stream(SpeechRequest("timeout", "你好"))
        try:
            await anext(stream)
        except RuntimeError as exc:
            assert "timed out" in str(exc)
        else:
            raise AssertionError("TTS worker timeout was not reported")
        finally:
            await stream.aclose()

    asyncio.run(scenario())


def test_subprocess_worker_starts_once_and_reuses_jsonl_stdin(tmp_path, caplog) -> None:
    calls_path = tmp_path / "calls.jsonl"
    pid_path = tmp_path / "worker.pid"
    worker = """
import base64
import json
import os
import pathlib
import sys

calls_path = pathlib.Path(sys.argv[1])
pathlib.Path(sys.argv[2]).write_text(str(os.getpid()), encoding="utf-8")
print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}), flush=True)
for line in sys.stdin:
    payload = json.loads(line)
    with calls_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\\n")
    print(json.dumps({
        "type": "audio",
        "protocol": "meapet.tts.jsonl",
        "version": 1,
        "request_id": payload["request_id"],
        "data": base64.b64encode(b"\\x00\\x00").decode("ascii"),
        "sample_rate": 24000,
        "channels": 1,
        "final": True,
    }), flush=True)
"""
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker, str(calls_path), str(pid_path)),
        shutdown_timeout_seconds=1.0,
    )

    async def scenario() -> None:
        first_health = await backend.start()
        assert first_health.available
        process = backend._process
        assert process is not None
        second_health = await backend.start()
        assert second_health.available
        assert backend._process is process
        first = [
            chunk
            async for chunk in backend.stream(
                SpeechRequest("first", "第一句", language="zh", mood="soft")
            )
        ]
        second = [
            chunk
            async for chunk in backend.stream(
                SpeechRequest("second", "第二句", language="zh", voice="voice-a")
            )
        ]
        assert first[-1].is_final
        assert second[-1].is_final
        assert backend._process is process
        await backend.aclose()
        assert backend._process is None

    with caplog.at_level("INFO"):
        asyncio.run(scenario())

    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    assert [call["request_id"] for call in calls] == ["first", "second"]
    assert all(call["type"] == "synthesize" for call in calls)
    assert all(call["streaming"] is True for call in calls)
    assert calls[1]["voice"] == "voice-a"
    assert all(
        "command" not in call and "cwd" not in call and "endpoint" not in call for call in calls
    )
    assert pid_path.read_text(encoding="utf-8").strip().isdigit()
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert '"event": "tts.backend.initialize"' in rendered
    assert '"event": "tts.request.start"' in rendered
    assert '"event": "tts.request.complete"' in rendered
    assert '"event": "tts.backend.unload"' in rendered
    assert event_fingerprint("first") in rendered
    assert event_fingerprint("second") in rendered
    assert '"correlation": "first"' not in rendered
    assert '"correlation": "second"' not in rendered


def test_subprocess_worker_ignores_late_done_after_final_audio() -> None:
    """常驻 worker 的 final 音频后迟到 done 不能污染下一次请求。"""

    worker = """
import base64
import json
import sys

print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({
        "type": "audio",
        "protocol": "meapet.tts.jsonl",
        "version": 1,
        "request_id": request["request_id"],
        "data": base64.b64encode(b"\\x00\\x00").decode("ascii"),
        "sample_rate": 24000,
        "channels": 1,
        "final": True,
    }), flush=True)
    print(json.dumps({
        "type": "done",
        "protocol": "meapet.tts.jsonl",
        "version": 1,
        "request_id": request["request_id"],
    }), flush=True)
"""
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker),
        startup_timeout_seconds=1.0,
        timeout_seconds=1.0,
    )

    async def scenario() -> None:
        first = [chunk async for chunk in backend.stream(SpeechRequest("final-one", "第一句"))]
        second = [chunk async for chunk in backend.stream(SpeechRequest("final-two", "第二句"))]
        assert [chunk.request_id for chunk in first] == ["final-one"]
        assert [chunk.request_id for chunk in second] == ["final-two"]
        assert first[-1].is_final is True
        assert second[-1].is_final is True
        await backend.aclose()

    asyncio.run(scenario())


def test_subprocess_start_waits_for_explicit_ready_handshake() -> None:
    worker = """
import json
import sys
import time

time.sleep(0.15)
print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}), flush=True)
for _line in sys.stdin:
    pass
"""
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker),
        startup_timeout_seconds=1.0,
    )

    async def scenario() -> float:
        started = time.monotonic()
        health = await backend.start()
        elapsed = time.monotonic() - started
        assert health.available is True
        await backend.aclose()
        return elapsed

    assert asyncio.run(scenario()) >= 0.12


@pytest.mark.parametrize(
    "worker",
    (
        "import json; print(json.dumps({'type':'error','message':'private'}), flush=True)",
        "pass",
        (
            "import json; print(json.dumps({'type':'ready',"
            "'protocol':'meapet.tts.jsonl','version':2,'streaming':True}), flush=True)"
        ),
    ),
)
def test_subprocess_start_rejects_error_eof_and_protocol_mismatch(worker: str) -> None:
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker),
        startup_timeout_seconds=1.0,
    )

    async def scenario() -> None:
        health = await backend.start()
        assert health.available is False
        assert health.message.startswith("worker initialization failed:")
        assert "private" not in health.message
        assert backend._process is None
        assert (await backend.health()).available is False
        await backend.aclose()

    asyncio.run(scenario())


def test_subprocess_start_drains_bounded_status_events_before_ready() -> None:
    worker = """
import json
import sys

for index in range(200):
    print(json.dumps({
        "type": "status",
        "protocol": "meapet.tts.jsonl",
        "version": 1,
        "step": index,
    }), flush=True)
print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}), flush=True)
for _line in sys.stdin:
    pass
"""
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker),
        startup_timeout_seconds=1.0,
    )

    async def scenario() -> None:
        assert (await backend.start()).available is True
        await backend.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "event",
    (
        {
            "type": "audio",
            "protocol": "meapet.tts.jsonl",
            "version": 1,
            "data": "AAA=",
            "sample_rate": 24000,
            "channels": 1,
            "final": True,
        },
        {
            "type": "done",
            "protocol": "meapet.tts.jsonl",
            "version": 1,
        },
        {
            "type": "error",
            "protocol": "meapet.tts.jsonl",
            "version": 1,
            "reason_code": "test_error",
        },
        {
            "type": "cancelled",
            "protocol": "meapet.tts.jsonl",
            "version": 1,
        },
    ),
)
def test_subprocess_stream_rejects_business_event_without_request_id(
    event: dict[str, object],
) -> None:
    worker = f"""
import json
import sys

print(json.dumps({{
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}}), flush=True)
json.loads(sys.stdin.readline())
print(json.dumps({event!r}), flush=True)
for _line in sys.stdin:
    pass
"""
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker),
        startup_timeout_seconds=1.0,
        timeout_seconds=1.0,
    )

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="request_id is required"):
            async for _chunk in backend.stream(SpeechRequest("strict-id", "你好")):
                pass
        await backend.aclose()

    asyncio.run(scenario())


def test_subprocess_stream_rejects_event_without_protocol_header() -> None:
    worker = """
import json
import sys

print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}), flush=True)
request = json.loads(sys.stdin.readline())
print(json.dumps({
    "type": "audio",
    "request_id": request["request_id"],
    "data": "AAA=",
    "sample_rate": 24000,
    "channels": 1,
    "final": True,
}), flush=True)
for _line in sys.stdin:
    pass
"""
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker),
        startup_timeout_seconds=1.0,
        timeout_seconds=1.0,
    )

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="invalid event"):
            async for _chunk in backend.stream(SpeechRequest("strict-protocol", "你好")):
                pass
        await backend.aclose()

    asyncio.run(scenario())


def test_subprocess_worker_environment_does_not_inherit_model_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_path = tmp_path / "worker-environment.json"
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-tts")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-reach-tts")
    monkeypatch.setenv("MEAPET_CHANNEL_API_KEY", "must-not-reach-tts")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/inherited/path")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/runtime/library/path")
    worker = """
import json
import os
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(os.environ)), encoding="utf-8")
print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}), flush=True)
for _line in sys.stdin:
    pass
"""
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker, str(environment_path)),
        startup_timeout_seconds=1.0,
    )

    async def scenario() -> None:
        assert (await backend.start()).available is True
        await backend.aclose()

    asyncio.run(scenario())
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    assert "OPENAI_API_KEY" not in environment
    assert "ANTHROPIC_AUTH_TOKEN" not in environment
    assert "MEAPET_CHANNEL_API_KEY" not in environment
    assert "/untrusted/inherited/path" not in environment["PYTHONPATH"]
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["LD_LIBRARY_PATH"] == "/runtime/library/path"
    assert set(environment) <= {
        "HOME",
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONUNBUFFERED",
    }


def test_subprocess_worker_stderr_is_bounded_and_fingerprinted(caplog) -> None:
    worker = (
        "import json,sys; "
        "print('token=private-value /home/private/model.ckpt'+'x'*5000,"
        "file=sys.stderr,flush=True); "
        "print(json.dumps({'type':'ready','protocol':'meapet.tts.jsonl',"
        "'version':1,'streaming':True}),flush=True); sys.stdin.read()"
    )
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker),
        max_line_bytes=1024,
        startup_timeout_seconds=1.0,
    )

    async def scenario() -> None:
        assert (await backend.start()).available is True
        await asyncio.sleep(0.05)
        await backend.aclose()

    with caplog.at_level(logging.DEBUG, logger="services.tts.backend"):
        asyncio.run(scenario())

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "ipc.worker.stderr" in rendered
    assert "stderr_fingerprint" in rendered
    assert "private-value" not in rendered
    assert "/home/private" not in rendered
    payloads = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "services.tts.backend" and record.getMessage().startswith("{")
    ]
    assert any(
        item.get("event") == "ipc.worker.stderr" and item.get("truncated") is True
        for item in payloads
    )
    starts = [item for item in payloads if item.get("event") == "tts.worker.ipc.start"]
    terminals = [item for item in payloads if item.get("event") == "tts.worker.ipc.complete"]
    assert starts and terminals
    assert starts[0]["operation"] == terminals[0]["operation"]


def test_process_liveness_probe_does_not_require_proc_on_non_linux(monkeypatch) -> None:
    import services.tts.backend as backend_module

    process = type("Process", (), {"pid": os.getpid()})()
    monkeypatch.setattr(backend_module.sys, "platform", "darwin")

    assert backend_module._posix_process_gone(process) is False


def test_process_liveness_probe_treats_unreadable_proc_as_unknown(monkeypatch) -> None:
    import services.tts.backend as backend_module

    process = type("Process", (), {"pid": os.getpid()})()
    monkeypatch.setattr(backend_module.sys, "platform", "linux")

    def unreadable_proc(*_args, **_kwargs):
        raise PermissionError("hidden proc")

    monkeypatch.setattr(Path, "read_text", unreadable_proc)

    assert backend_module._posix_process_gone(process) is False


def test_subprocess_close_sends_stdin_eof_before_bounded_shutdown(tmp_path) -> None:
    eof_path = tmp_path / "stdin-eof"
    worker = """
import pathlib
import json
import sys

print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}), flush=True)
for _line in sys.stdin:
    pass
pathlib.Path(sys.argv[1]).write_text("closed", encoding="utf-8")
"""
    backend = SubprocessTTSBackend(
        (sys.executable, "-c", worker, str(eof_path)),
        shutdown_timeout_seconds=1.0,
    )

    async def scenario() -> None:
        health = await backend.start()
        assert health.available
        await backend.aclose()

    asyncio.run(scenario())
    assert eof_path.read_text(encoding="utf-8") == "closed"


def test_subprocess_close_serializes_with_inflight_start(tmp_path, monkeypatch) -> None:
    worker = """
import json
import sys

print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}), flush=True)
for _line in sys.stdin:
    pass
"""
    backend = SubprocessTTSBackend((sys.executable, "-c", worker))
    create_entered = asyncio.Event()
    create_release = asyncio.Event()
    original_create = asyncio.create_subprocess_exec

    async def delayed_create(*args, **kwargs):
        create_entered.set()
        await create_release.wait()
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_create)

    async def scenario() -> None:
        start_task = asyncio.create_task(backend.start())
        await create_entered.wait()
        close_task = asyncio.create_task(backend.aclose())
        await asyncio.sleep(0)
        assert close_task.done() is False
        create_release.set()
        start_health, _ = await asyncio.gather(start_task, close_task)
        assert start_health.available is True
        assert backend._closed is True
        assert backend._process is None
        assert (await backend.health()).available is False

    asyncio.run(scenario())


def test_coordinator_start_initializes_backend_once_and_close_unloads_it() -> None:
    class Backend:
        def __init__(self) -> None:
            self.starts = 0
            self.closes = 0

        async def start(self) -> EngineHealth:
            self.starts += 1
            return EngineHealth("lifecycle", True, "ready")

        async def health(self) -> EngineHealth:
            return EngineHealth("lifecycle", True, "ready")

        async def stream(self, request: SpeechRequest):
            if False:
                yield SpeechChunk(request.request_id, b"")

        async def aclose(self) -> None:
            self.closes += 1

    backend = Backend()
    coordinator = TTSCoordinator(backend)

    async def scenario() -> None:
        first = await coordinator.start()
        second = await coordinator.start()
        assert first.available and second.available
        await coordinator.aclose()

    asyncio.run(scenario())
    assert backend.starts == 1
    assert backend.closes == 1


def test_coordinator_concurrent_start_is_single_flight() -> None:
    class Backend:
        def __init__(self) -> None:
            self.starts = 0

        async def start(self) -> EngineHealth:
            self.starts += 1
            await asyncio.sleep(0.02)
            return EngineHealth("lifecycle", True, "ready")

        async def health(self) -> EngineHealth:
            return EngineHealth("lifecycle", True, "ready")

        async def stream(self, request: SpeechRequest):
            if False:
                yield SpeechChunk(request.request_id, b"")

        async def aclose(self) -> None:
            return None

    backend = Backend()
    coordinator = TTSCoordinator(backend)

    async def scenario() -> None:
        first, second = await asyncio.gather(coordinator.start(), coordinator.start())
        assert first.available and second.available
        await coordinator.aclose()

    asyncio.run(scenario())
    assert backend.starts == 1


def test_coordinator_close_waits_for_inflight_start_before_unloading() -> None:
    class Backend:
        def __init__(self) -> None:
            self.start_entered = asyncio.Event()
            self.start_release = asyncio.Event()
            self.events: list[str] = []

        async def start(self) -> EngineHealth:
            self.events.append("start_entered")
            self.start_entered.set()
            await self.start_release.wait()
            self.events.append("start_completed")
            return EngineHealth("lifecycle", True, "ready")

        async def health(self) -> EngineHealth:
            return EngineHealth("lifecycle", True, "ready")

        async def stream(self, request: SpeechRequest):
            if False:
                yield SpeechChunk(request.request_id, b"")

        async def aclose(self) -> None:
            self.events.append("closed")

    async def scenario() -> list[str]:
        backend = Backend()
        coordinator = TTSCoordinator(backend)
        start_task = asyncio.create_task(coordinator.start())
        await backend.start_entered.wait()
        close_task = asyncio.create_task(coordinator.aclose())
        await asyncio.sleep(0)
        assert backend.events == ["start_entered"]
        backend.start_release.set()
        await asyncio.gather(start_task, close_task)
        assert coordinator.closed is True
        return backend.events

    assert asyncio.run(scenario()) == ["start_entered", "start_completed", "closed"]


def test_coordinator_cancelled_close_waiter_does_not_cancel_shared_cleanup() -> None:
    class Backend:
        def __init__(self) -> None:
            self.close_entered = asyncio.Event()
            self.close_release = asyncio.Event()
            self.closes = 0

        async def health(self) -> EngineHealth:
            return EngineHealth("lifecycle", True, "ready")

        async def stream(self, request: SpeechRequest):
            if False:
                yield SpeechChunk(request.request_id, b"")

        async def aclose(self) -> None:
            self.closes += 1
            self.close_entered.set()
            await self.close_release.wait()

    async def scenario() -> None:
        backend = Backend()
        coordinator = TTSCoordinator(backend)
        first_waiter = asyncio.create_task(coordinator.aclose())
        await backend.close_entered.wait()
        assert coordinator.closed is False

        first_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter
        assert coordinator.closed is False

        second_waiter = asyncio.create_task(coordinator.aclose())
        await asyncio.sleep(0)
        assert second_waiter.done() is False
        backend.close_release.set()
        await second_waiter
        assert coordinator.closed is True
        assert backend.closes == 1

    asyncio.run(scenario())


def test_coordinator_backend_replace_waits_for_inflight_start() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Backend:
        def __init__(self, engine: str) -> None:
            self.engine = engine

        async def start(self) -> EngineHealth:
            entered.set()
            await release.wait()
            return EngineHealth(self.engine, False, "old unavailable")

        async def health(self) -> EngineHealth:
            return EngineHealth(self.engine, True, "ready")

        async def stream(self, request: SpeechRequest):
            if False:
                yield SpeechChunk(request.request_id, b"")

        async def aclose(self) -> None:
            return None

    old_backend = Backend("old")
    new_backend = Backend("new")
    coordinator = TTSCoordinator(old_backend)

    async def scenario() -> None:
        start_task = asyncio.create_task(coordinator.start())
        await entered.wait()
        replace_task = asyncio.create_task(
            coordinator.replace_backend(new_backend, initialized=True)
        )
        await asyncio.sleep(0)
        assert replace_task.done() is False
        release.set()
        await start_task
        previous = await replace_task
        assert previous is old_backend
        assert coordinator.backend is new_backend
        assert coordinator.started is True
        await coordinator.aclose()

    asyncio.run(scenario())


def test_tts_logs_request_metadata_without_text_or_options(caplog) -> None:
    class Backend:
        def __init__(self) -> None:
            self.request_id = ""

        async def health(self) -> EngineHealth:
            return EngineHealth("log-test", True, "ready")

        async def stream(self, request: SpeechRequest):
            self.request_id = request.request_id
            yield SpeechChunk(
                request.request_id,
                b"\x00\x00",
                sample_rate=24000,
                channels=1,
                is_final=True,
            )

    backend = Backend()
    coordinator = TTSCoordinator(backend)
    context = ConversationContext("provider", "session", "turn", 1)
    secret_key = "sk-private-tts-key"
    secret_text = f"PRIVATE_TTS_BODY_9417 {secret_key}"

    async def scenario() -> None:
        await coordinator.enqueue_text(
            context,
            secret_text,
            language="zh",
            mood="soft",
            voice="voice-a",
            flush=True,
        )
        await coordinator.aclose()

    with caplog.at_level("INFO"):
        asyncio.run(scenario())

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert '"event": "tts.synthesis.start"' in rendered
    assert '"event": "tts.synthesis.complete"' in rendered
    assert '"language": "zh"' in rendered
    assert '"voice": "voice-a"' in rendered
    assert event_fingerprint(backend.request_id) in rendered
    assert backend.request_id not in rendered
    assert secret_text not in rendered
    assert secret_key not in rendered


def test_tts_worker_ipc_cancellation_uses_info_terminal_event(caplog) -> None:
    backend = SubprocessTTSBackend((sys.executable, "-c", "pass"))

    with caplog.at_level(logging.INFO, logger="services.tts.backend"):
        backend._log_ipc(
            "start",
            "cancelled",
            reason_code="startup_cancelled",
            level=logging.INFO,
        )

    records = [
        record
        for record in caplog.records
        if '"event": "tts.worker.ipc.cancelled"' in record.getMessage()
    ]
    assert len(records) == 1
    payload = json.loads(records[0].getMessage())
    assert payload["status"] == "cancelled"
    assert payload["reason_code"] == "startup_cancelled"
    assert records[0].levelno == logging.INFO
    assert '"event": "tts.worker.ipc.failed"' not in records[0].getMessage()


def test_text_only_debug_log_fingerprints_request_id(caplog) -> None:
    backend = TextOnlyBackend()
    request = SpeechRequest("private-request-id", "不应记录的正文")

    async def scenario() -> None:
        assert [chunk async for chunk in backend.stream(request)] == []

    with caplog.at_level("DEBUG"):
        asyncio.run(scenario())

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert '"event": "tts.request.skipped"' in rendered
    assert event_fingerprint(request.request_id) in rendered
    assert request.request_id not in rendered
    assert request.text not in rendered
