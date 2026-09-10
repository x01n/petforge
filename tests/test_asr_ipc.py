from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from core.asr import ASR_IPC_SPEC, ASRHealth, TranscriptionRequest, TranscriptionResult
from services.asr import ASRService
from services.asr import worker as asr_worker

_FAKE_WORKER = r"""
import json,sys,threading,time
protocol='meapet.asr.jsonl'
lock=threading.Lock()
active={}
def emit(kind,request_id='',**fields):
    value={'type':kind,'protocol':protocol,'version':1,**fields}
    if request_id: value['request_id']=request_id
    with lock:
        print(json.dumps(value,separators=(',',':')),flush=True)
def finish(request_id):
    time.sleep(.25)
    if active.pop(request_id,False):
        emit('transcript',request_id,text='本地转写',language='zh',confidence=None,
             confidence_available=False,duration_ms=12)
emit('ready',backend='sensevoice',model_loaded=False)
for line in sys.stdin:
    message=json.loads(line)
    kind=message['type']; request_id=message.get('request_id','')
    if kind=='startup': emit('started',request_id,accepted=True)
    elif kind=='load_model': emit('model_loaded',request_id,loaded=True)
    elif kind=='health': emit('health',request_id,ready=True,model_loaded=True)
    elif kind=='transcribe':
        if 'path' in message or any(key.endswith('_path') for key in message):
            emit('error',request_id,reason_code='privacy_boundary_failed')
        else:
            active[request_id]=True
            threading.Thread(target=finish,args=(request_id,),daemon=True).start()
    elif kind=='cancel':
        cancelled=bool(active.pop(request_id,False))
        emit('cancelled',request_id,cancelled=cancelled)
        if cancelled: break
    elif kind=='shutdown':
        emit('shutdown',request_id,stopped=True)
        break
"""

_SLOW_LOAD_WORKER = _FAKE_WORKER.replace(
    "elif kind=='load_model': emit('model_loaded',request_id,loaded=True)",
    "elif kind=='load_model': time.sleep(5); emit('model_loaded',request_id,loaded=True)",
)


def _wav_bytes() -> bytes:
    return b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 32


def _service(worker: str = _FAKE_WORKER) -> ASRService:
    return ASRService(
        (sys.executable, "-u", "-c", worker),
        enabled=True,
        backend="sensevoice",
        model_name="SenseVoiceSmall",
        device="cpu",
        language="auto",
        timeout_seconds=2.0,
        startup_timeout_seconds=2.0,
        max_audio_bytes=4096,
    )


def test_asr_protocol_declares_complete_lifecycle() -> None:
    assert ASR_IPC_SPEC.protocol == "meapet.asr.jsonl"
    assert ASR_IPC_SPEC.version == 1
    assert ASR_IPC_SPEC.commands == frozenset(
        {"startup", "load_model", "health", "transcribe", "cancel", "shutdown"}
    )
    assert {"ready", "model_loaded", "transcript", "cancelled", "shutdown"}.issubset(
        ASR_IPC_SPEC.events
    )


def test_asr_request_accepts_only_bounded_audio_contracts() -> None:
    request = TranscriptionRequest(_wav_bytes(), "wav", 16000, 1, "zh")
    assert request.audio_format == "wav"
    assert request.language == "zh"
    with pytest.raises(ValueError, match="WAV"):
        TranscriptionRequest(b"not-wav", "wav")
    with pytest.raises(ValueError, match="frame aligned"):
        TranscriptionRequest(b"\x00", "pcm_s16le", channels=1)
    with pytest.raises(ValueError, match="language"):
        TranscriptionRequest(_wav_bytes(), language="Chinese")


def test_asr_worker_rejects_encoded_audio_limit_before_base64_decode(monkeypatch) -> None:
    byte_limit = 1024
    encoded_limit = ((byte_limit + 2) // 3) * 4
    worker = asr_worker.SenseVoiceWorker(
        backend="sensevoice",
        model_path="unused-in-boundary-test",
        device="cpu",
        language="auto",
        max_audio_bytes=byte_limit,
    )
    worker.model = object()  # type: ignore[assignment]
    worker._active_request_id = "oversized-request"
    errors: list[tuple[str, str]] = []

    def reject_decode(*_args, **_kwargs):
        raise AssertionError("base64 decoder must not receive oversized input")

    monkeypatch.setattr(asr_worker.base64, "b64decode", reject_decode)
    monkeypatch.setattr(
        asr_worker,
        "_error",
        lambda request_id, reason_code: errors.append((request_id, reason_code)),
    )
    worker._run_transcription(
        "oversized-request",
        {
            "audio_base64": "A" * (encoded_limit + 1),
            "audio_format": "wav",
            "sample_rate": 16000,
            "channels": 1,
            "language": "zh",
        },
        threading.Event(),
    )

    assert errors == [("oversized-request", "audio_limit")]
    assert worker._active_request_id == ""


def test_asr_result_requires_truthful_confidence_availability() -> None:
    result = TranscriptionResult("request-1", "你好", "zh", None, False, 12)
    assert result.public()["confidence_available"] is False
    with pytest.raises(ValueError, match="availability"):
        TranscriptionResult("request-2", "你好", "zh", None, True, 12)


def test_asr_service_uses_shared_ipc_without_paths_or_text_diagnostics() -> None:
    async def scenario() -> None:
        service = _service()
        try:
            health = await service.start()
            assert health.ready is True
            result = await service.transcribe(_wav_bytes(), language="zh")
            assert result.text == "本地转写"
            assert result.language == "zh"
            assert result.confidence is None
            diagnostics = service.diagnostics()
            rendered = repr(diagnostics)
            assert diagnostics["status"] == "ready"
            assert diagnostics["ipc"]["running"] is True
            assert "本地转写" not in rendered
            assert "python" not in rendered.lower()
            assert "path" not in rendered.lower()
        finally:
            await service.aclose()

    asyncio.run(scenario())


def test_asr_cancel_stops_non_interruptible_worker_and_recovers_cleanly() -> None:
    async def scenario() -> None:
        service = _service()
        try:
            await service.start()
            request_id = "cancel-request-1"
            task = asyncio.create_task(
                service.transcribe(_wav_bytes(), language="zh", request_id=request_id)
            )
            for _index in range(50):
                if service.has_active_tasks():
                    break
                await asyncio.sleep(0.01)
            health = await asyncio.wait_for(service.health(probe=True), timeout=0.05)
            assert health.ready is True
            with pytest.raises(RuntimeError, match="already running"):
                await service.transcribe(_wav_bytes(), language="zh")
            assert await service.cancel(request_id) is True
            with pytest.raises(asyncio.CancelledError):
                await task
            assert service.running is False
            recovered = await service.transcribe(_wav_bytes(), language="zh")
            assert recovered.text == "本地转写"
            assert service.running is True
        finally:
            await service.aclose()

    asyncio.run(scenario())


def test_asr_start_failure_is_degraded_not_exception() -> None:
    async def scenario() -> None:
        service = ASRService(
            ("/path/that/does/not/exist/python", "-m", "services.asr.worker"),
            enabled=True,
            backend="sensevoice",
            model_name="SenseVoiceSmall",
            device="cpu",
            language="auto",
            timeout_seconds=0.5,
            startup_timeout_seconds=0.5,
            max_audio_bytes=4096,
        )
        try:
            health = await service.start()
            assert health.available is False
            assert health.status == "unavailable"
        finally:
            await service.aclose()

    asyncio.run(scenario())


def test_asr_health_probe_returns_cached_loading_state_without_waiting_for_start_lock() -> None:
    async def scenario() -> None:
        service = _service()
        service._health = ASRHealth(
            "loading", False, "sensevoice", "SenseVoiceSmall", False, "cpu", "auto"
        )
        try:
            async with service._start_lock:
                health = await asyncio.wait_for(service.health(probe=True), timeout=0.05)
            assert health.status == "loading"
            assert health.available is False
        finally:
            await service.aclose()

    asyncio.run(scenario())


def test_asr_close_cancels_pending_model_load_without_waiting_for_request_timeout() -> None:
    async def scenario() -> None:
        service = _service(_SLOW_LOAD_WORKER)
        task = asyncio.create_task(
            service.transcribe(_wav_bytes(), language="zh", request_id="pending-load")
        )
        for _index in range(100):
            if service._pending_request_id == "pending-load":
                break
            await asyncio.sleep(0.01)
        await asyncio.wait_for(service.aclose(), timeout=1.0)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service.running is False
        assert service.closed is True

    asyncio.run(scenario())


def test_asr_cancelled_close_waiter_does_not_cancel_shared_worker_cleanup() -> None:
    async def scenario() -> None:
        service = _service()
        health = await service.start()
        assert health.ready is True
        process = service._client.process
        assert process is not None
        close_entered = asyncio.Event()
        close_release = asyncio.Event()
        original_shutdown = service._client.shutdown
        shutdown_calls = 0

        async def gated_shutdown() -> None:
            nonlocal shutdown_calls
            shutdown_calls += 1
            close_entered.set()
            await close_release.wait()
            await original_shutdown()

        service._client.shutdown = gated_shutdown  # type: ignore[method-assign]
        first_waiter = asyncio.create_task(service.aclose())
        await close_entered.wait()
        assert service.closed is False
        assert service.running is True

        first_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter
        assert service.closed is False
        assert service.running is True

        second_waiter = asyncio.create_task(service.aclose())
        await asyncio.sleep(0)
        assert second_waiter.done() is False
        close_release.set()
        await second_waiter
        assert service.closed is True
        assert service.running is False
        assert service._client.process is None
        assert shutdown_calls == 1

    asyncio.run(scenario())
