"""第 9 轮方向 1（IPC 与数据层并发治理）定向回归测试。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.ipc import IPCProtocolSpec
from core.memory_semantic import (
    SemanticMemoryError,
    SemanticModelSpec,
    SentenceTransformerProcess,
    SentenceTransformerProcessSettings,
)
from core.tts.contracts import SpeechRequest
from db import ApiCallAuditRecord, ApiCallAuditRepository, Database, sanitize_audit_value
from db.scheduler_repository import SchedulerRepository
from services.ipc import IPCProcessError, JSONLSubprocessClient
from services.tools.executor import ToolExecutionService
from services.tools.permissions import PermissionService
from services.tools.registry import ToolRegistry
from services.tools.types import RiskLevel, ToolCallContext, ToolKind, ToolSpec
from services.tts import backpress
from services.tts.backend import SubprocessTTSBackend
from services.tts.backpress import RequestBacklogError

_READY_WORKER = r"""
import json, sys
print(json.dumps({
    'type': 'ready', 'protocol': 'meapet.test.jsonl', 'version': 1,
}), flush=True)
for raw in sys.stdin:
    message = json.loads(raw)
    if message['type'] == 'shutdown':
        print(json.dumps({
            'type': 'shutdown', 'protocol': 'meapet.test.jsonl', 'version': 1,
            'request_id': message.get('request_id', ''),
        }), flush=True)
        break
"""

_EXIT_WORKER = "import sys\nsys.exit(2)"

_STALL_TTS_WORKER = r"""
import json, sys, time
print(json.dumps({
    'type': 'ready', 'protocol': 'meapet.tts.jsonl', 'version': 1,
    'streaming': True,
    'capabilities': ['cancel', 'shutdown', 'streaming_synthesis'],
}), flush=True)
for raw in sys.stdin:
    message = json.loads(raw)
    if message['type'] == 'synthesize':
        while True:
            time.sleep(0.1)
    elif message['type'] == 'shutdown':
        print(json.dumps({
            'type': 'shutdown', 'protocol': 'meapet.tts.jsonl', 'version': 1,
            'request_id': message.get('request_id', ''), 'clean': True,
        }), flush=True)
        break
"""

_UNCLEAN_SHUTDOWN_WORKER = r"""
import json, sys
print(json.dumps({
    'type': 'ready', 'protocol': 'meapet.tts.jsonl', 'version': 1,
    'streaming': True,
    'capabilities': ['cancel', 'shutdown', 'streaming_synthesis'],
}), flush=True)
for raw in sys.stdin:
    message = json.loads(raw)
    if message['type'] == 'shutdown':
        print(json.dumps({
            'type': 'shutdown', 'protocol': 'meapet.tts.jsonl', 'version': 1,
            'request_id': message.get('request_id', ''), 'clean': False,
        }), flush=True)
        break
"""

_SILENT_TTS_WORKER = r"""
import json, sys
print(json.dumps({
    'type': 'ready', 'protocol': 'meapet.tts.jsonl', 'version': 1,
    'streaming': True,
    'capabilities': ['cancel', 'shutdown', 'streaming_synthesis'],
}), flush=True)
for raw in sys.stdin:
    message = json.loads(raw)
    if message['type'] == 'shutdown':
        print(json.dumps({
            'type': 'shutdown', 'protocol': 'meapet.tts.jsonl', 'version': 1,
            'request_id': message.get('request_id', ''), 'clean': True,
        }), flush=True)
        break
"""


def _slow_tts_worker(interval: float, count: int) -> str:
    """按固定间隔产生合法 PCM 分片的 TTS worker（data 'AAE=' 解出 2 字节）。"""

    return f"""
import json, sys, time
print(json.dumps({{
    'type': 'ready', 'protocol': 'meapet.tts.jsonl', 'version': 1,
    'streaming': True,
    'capabilities': ['cancel', 'shutdown', 'streaming_synthesis'],
}}), flush=True)
for raw in sys.stdin:
    message = json.loads(raw)
    if message['type'] == 'synthesize':
        request_id = message.get('request_id', '')
        for index in range({count}):
            print(json.dumps({{
                'type': 'audio', 'protocol': 'meapet.tts.jsonl', 'version': 1,
                'request_id': request_id, 'data': 'AAE=', 'final': index == {count} - 1,
                'sample_rate': 24000, 'channels': 1, 'sample_format': 's16le',
            }}), flush=True)
            if index < {count} - 1:
                time.sleep({interval!r})
    elif message['type'] == 'shutdown':
        print(json.dumps({{
            'type': 'shutdown', 'protocol': 'meapet.tts.jsonl', 'version': 1,
            'request_id': message.get('request_id', ''), 'clean': True,
        }}), flush=True)
        break
"""


def _tts_backend(worker: str, **kwargs) -> SubprocessTTSBackend:
    return SubprocessTTSBackend(
        (sys.executable, "-u", "-c", worker),
        timeout_seconds=kwargs.pop("timeout_seconds", 5.0),
        startup_timeout_seconds=kwargs.pop("startup_timeout_seconds", 5.0),
        shutdown_timeout_seconds=kwargs.pop("shutdown_timeout_seconds", 0.5),
        **kwargs,
    )


def _test_spec() -> IPCProtocolSpec:
    return IPCProtocolSpec(
        "meapet.test.jsonl",
        1,
        frozenset({"shutdown"}),
        frozenset({"ready", "shutdown", "error"}),
        ("shutdown",),
    )


def test_a_side_effect_lock_survives_cross_event_loop_calls() -> None:
    """不同线程/新事件循环调用副作用工具不抛跨循环 RuntimeError。"""

    running = 0
    maximum = 0
    started = threading.Event()
    release = threading.Event()

    def side_handler(_arguments, _context):
        nonlocal running, maximum
        running += 1
        maximum = max(maximum, running)
        started.set()
        release.wait(5.0)
        running -= 1
        return {"status": "ok"}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:side_effect",
            "side effect",
            {"type": "object", "properties": {}, "additionalProperties": False},
            side_handler,
            RiskLevel.LOW,
            kind=ToolKind.SYSTEM,
            read_only=False,
        )
    )
    executor = ToolExecutionService(registry, PermissionService(auto_allow_low_risk=True))
    context = ToolCallContext("profile", "session", "side-turn")
    outcomes: list[object] = []

    def run_in_thread(index: int) -> None:
        loop = asyncio.new_event_loop()
        try:
            outcomes.append(
                loop.run_until_complete(
                    executor.execute(
                        call_id=f"side-{index}",
                        identity="system:side_effect",
                        arguments={},
                        context=context,
                    )
                )
            )
        finally:
            loop.close()

    threads = [threading.Thread(target=run_in_thread, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    started.wait(5.0)
    release.set()
    for thread in threads:
        thread.join(5.0)
    assert not any(thread.is_alive() for thread in threads)
    assert maximum == 1
    assert len(outcomes) == 2
    assert all(getattr(outcome, "status", None) == "completed" for outcome in outcomes)


def test_a_read_only_semaphore_foreign_loop_is_rejected() -> None:
    """只读信号量被占用且绑定主循环时，跨循环调用返回明确拒绝结果。"""

    release = asyncio.Event()

    async def handler(_arguments, _context):
        await release.wait()
        return {"status": "ok"}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:read",
            "read tool",
            {"type": "object", "properties": {}, "additionalProperties": False},
            handler,
            RiskLevel.LOW,
            kind=ToolKind.SYSTEM,
            read_only=True,
        )
    )
    executor = ToolExecutionService(registry, PermissionService(auto_allow_low_risk=True))
    context = ToolCallContext("profile", "session", "read-turn")
    foreign_results: list[object] = []

    def foreign_loop() -> None:
        loop = asyncio.new_event_loop()
        try:
            foreign_results.append(
                loop.run_until_complete(
                    executor.execute(
                        call_id="foreign",
                        identity="system:read",
                        arguments={},
                        context=context,
                    )
                )
            )
        finally:
            loop.close()

    async def scenario() -> None:
        # 四个只读调用在主循环占满 4 格信号量
        occupied = [
            asyncio.create_task(
                executor.execute(
                    call_id=f"hold-{index}",
                    identity="system:read",
                    arguments={},
                    context=context,
                )
            )
            for index in range(4)
        ]
        await asyncio.sleep(0.05)
        thread = threading.Thread(target=foreign_loop, daemon=True)
        thread.start()
        thread.join(5.0)
        assert not thread.is_alive()
        assert foreign_results
        foreign = foreign_results[0]
        assert getattr(foreign, "status", None) == "failed"
        assert "another event loop" in str(getattr(foreign, "content", {}).get("error", ""))
        release.set()
        finished = await asyncio.gather(*occupied)
        assert all(outcome.status == "completed" for outcome in finished)

    asyncio.run(scenario())


def test_b_tts_second_stream_is_rejected_as_busy(monkeypatch) -> None:
    """长请求挂起时第二次调用收到明确否决而不是永久挂起。"""

    monkeypatch.setattr(backpress, "REQUEST_QUEUE_WAIT_SECONDS", 0.2)
    backend = _tts_backend(_STALL_TTS_WORKER)

    async def scenario() -> None:
        health = await backend.start()
        assert health.available is True
        first = backend.stream(SpeechRequest("stall-1", "占住", language="zh"))
        stream_task = asyncio.create_task(anext(first))
        deadline = time.monotonic() + 5.0
        while not backend._request_lock.locked() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert backend._request_lock.locked()
        started = time.monotonic()
        with pytest.raises(RequestBacklogError, match="queue is unavailable"):
            async for _chunk in backend.stream(SpeechRequest("stall-2", "排队", language="zh")):
                pass
        assert time.monotonic() - started < 2.0
        stream_task.cancel()
        await asyncio.gather(stream_task, return_exceptions=True)
        await first.aclose()
        await backend.aclose()

    asyncio.run(scenario())


def test_c_scheduler_read_methods_take_database_lock() -> None:
    """调度仓储读方法在与写并发时保持返回形状。"""

    repository = SchedulerRepository(Database(":memory:"))
    repository.save_task(
        {
            "task_id": "task-1",
            "name": "晚间播报",
            "expression": "0 21 * * *",
            "action": {"type": "speak", "text": "晚安"},
            "owner": "core",
            "next_run_at": 2.0,
            "enabled": True,
            "run_count": 0,
            "metadata": {},
        }
    )
    assert len(repository.load_tasks()) == 1
    assert repository.load_triggers() == ()
    assert repository.list_runs(limit=10) == ()

    def writer() -> None:
        for _ in range(30):
            repository.save_task(
                {
                    "task_id": "task-1",
                    "name": "晚间播报",
                    "expression": "0 21 * * *",
                    "action": {"type": "speak", "text": "晚安"},
                    "owner": "core",
                    "next_run_at": 2.0,
                    "enabled": True,
                    "run_count": 0,
                    "metadata": {},
                }
            )
            repository.record_run(
                kind="task",
                item_id="task-1",
                owner="core",
                started_at=1.0,
                finished_at=2.0,
                status="ok",
            )

    def reader() -> None:
        for _ in range(30):
            assert len(repository.load_tasks()) == 1
            assert repository.load_triggers() == ()
            repository.list_runs(limit=10)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)
    assert not any(thread.is_alive() for thread in threads)
    assert len(repository.load_tasks()) == 1
    assert repository.list_runs(limit=10)


def test_d_audit_auto_prune_converges_and_indexes_exist() -> None:
    """写入路径自动修剪且幂等索引存在，旧库重复打开不破坏表。"""

    repository = ApiCallAuditRepository(Database(":memory:"))
    repository._write_count = 96
    prune_calls: list[int] = []
    original_prune = repository.prune

    def spying_prune(*, max_records: int = 10_000) -> int:
        prune_calls.append(max_records)
        return original_prune(max_records=max_records)

    repository.prune = spying_prune  # type: ignore[method-assign]
    for index in range(6):
        repository.save(
            ApiCallAuditRecord(
                request_id=f"auto-prune-{index}",
                started_at=float(index),
                created_at=1.0,
                updated_at=1.0,
            )
        )
    assert prune_calls

    again = ApiCallAuditRepository(repository._database)
    index_rows = repository._database.connection.execute(
        f"PRAGMA index_list({repository._TABLE})"
    ).fetchall()
    index_names = {str(row["name"]) for row in index_rows}
    assert "idx_api_call_audits_status" in index_names
    assert "idx_api_call_audits_channel" in index_names
    assert again.count() >= 1


def test_l_sanitize_audit_value_redacts_expanded_secret_fields() -> None:
    """白名单扩充后的敏感字段被遮蔽，公共字段保留。"""

    payload = {
        "client_secret": "hidden-client",
        "private_key": "hidden-key",
        "session_key": "hidden-session",
        "subscription_key": "hidden-sub",
        "webhook_secret": "hidden-webhook",
        "webhook-secret": "hidden-dash",
        "model": "public-field",
    }
    cleaned = sanitize_audit_value(payload)
    assert isinstance(cleaned, dict)
    assert cleaned["model"] == "public-field"
    for key in (
        "client_secret",
        "private_key",
        "session_key",
        "subscription_key",
        "webhook_secret",
        "webhook-secret",
    ):
        assert cleaned[key] == "[redacted]"


def test_i_slow_chunk_stream_resets_idle_deadline() -> None:
    """慢分片流按空闲超时语义存活，超预算静默仍被枪毙。"""

    backend = _tts_backend(_slow_tts_worker(0.35, 7), timeout_seconds=0.6)

    async def slow_stream() -> None:
        health = await backend.start()
        assert health.available is True
        chunks = [
            chunk.data
            async for chunk in backend.stream(SpeechRequest("slow", "慢分片", language="zh"))
        ]
        assert len(chunks) == 7
        assert sum(len(chunk) for chunk in chunks) >= 2
        await backend.aclose()

    asyncio.run(slow_stream())

    silent = _tts_backend(_SILENT_TTS_WORKER, timeout_seconds=0.4)

    async def silent_stream() -> None:
        health = await silent.start()
        assert health.available is True
        with pytest.raises(RuntimeError, match="timed out"):
            async for _chunk in silent.stream(SpeechRequest("silent", "静默", language="zh")):
                pass
        await silent.aclose()

    asyncio.run(silent_stream())


def test_h_unclean_shutdown_message_forces_process_recycle(monkeypatch) -> None:
    """worker 回 clean=False 时关闭走强制回收（terminate 兜底）。"""

    backend = _tts_backend(_UNCLEAN_SHUTDOWN_WORKER)
    captured: dict[str, bool] = {}

    async def scenario() -> None:
        real_dispose = backend._dispose_process

        async def recording_dispose(process, *, graceful: bool = False) -> None:
            captured["graceful"] = graceful
            await real_dispose(process, graceful=graceful)

        monkeypatch.setattr(backend, "_dispose_process", recording_dispose)
        health = await backend.start()
        assert health.available is True
        await backend.aclose()
        assert backend._closed is True
        assert backend._process is None

    asyncio.run(scenario())
    assert captured.get("graceful") is False


def test_e_start_failure_counts_backoff_and_success_resets() -> None:
    """连续启动失败累计退避，成功启动清零。"""

    bad = JSONLSubprocessClient(
        _test_spec(),
        (sys.executable, "-u", "-c", _EXIT_WORKER),
        startup_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.5,
        backend="test-backoff-bad",
    )

    async def fail_once() -> None:
        with pytest.raises(IPCProcessError):
            await bad.start()

    asyncio.run(fail_once())
    assert bad._start_failure_count == 1
    assert bad._backoff_remaining == pytest.approx(2.0)
    asyncio.run(bad.shutdown())

    good = JSONLSubprocessClient(
        _test_spec(),
        (sys.executable, "-u", "-c", _READY_WORKER),
        startup_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.5,
        backend="test-backoff-good",
    )
    good._start_failure_count = 3
    good._backoff_remaining = 0.01

    async def succeed() -> None:
        ready = await good.start()
        assert ready.type == "ready"
        assert good._start_failure_count == 0
        assert good._backoff_remaining == 0.0
        await good.shutdown()

    asyncio.run(succeed())


def test_f_startup_events_is_bounded_deque() -> None:
    """startup_events 用 deque(maxlen=64) 且 clear/append 接口不变。"""

    import collections

    client = JSONLSubprocessClient(
        _test_spec(),
        (sys.executable, "-u", "-c", _READY_WORKER),
        startup_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.5,
        backend="test-deque",
    )
    assert isinstance(client.startup_events, collections.deque)
    assert client.startup_events.maxlen == 64
    client.startup_events.clear()
    for _ in range(80):
        client.startup_events.append(object())
    assert len(client.startup_events) == 64
    client.startup_events.clear()


def test_g_dispose_awaits_stdin_wait_closed() -> None:
    """_dispose 与 TTS 后端对齐：stdin 关闭后等待 wait_closed()。"""

    waited: list[int] = []

    class FakeStdin:
        def __init__(self) -> None:
            self.closed = False

        def is_closing(self) -> bool:
            return self.closed

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            if not self.closed:
                raise RuntimeError("stdin not closed yet")
            waited.append(1)

    async def scenario() -> None:
        client = JSONLSubprocessClient(
            _test_spec(),
            (sys.executable, "-u", "-c", _READY_WORKER),
            startup_timeout_seconds=1.0,
            shutdown_timeout_seconds=0.5,
            backend="test-wait-closed",
        )
        fake_stdin = FakeStdin()
        fake_process = SimpleNamespace(
            stdin=fake_stdin,
            returncode=0,
            pid=None,
            stdout=None,
            stderr=None,
        )
        client._process = fake_process  # type: ignore[assignment]
        client._group_id = None
        client._wait_task = None
        client._stderr_task = None
        await client._dispose(graceful=True)
        assert client._process is None

    asyncio.run(scenario())
    assert waited == [1]


def test_j_memory_request_terminates_only_on_protocol_faults() -> None:
    """CancelledError 保留进程，协议/轮询故障继续终止。"""

    settings = SentenceTransformerProcessSettings(
        model_path=Path("/nonexistent/model"),
        model_id="local-test-model",
        model_revision="r1",
        dimension=8,
        batch_size=2,
        timeout_seconds=1.0,
        startup_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.5,
        max_text_chars=64,
        max_text_bytes=128,
    )
    provider = SentenceTransformerProcess(settings)

    class FakeProcess:
        returncode = None
        stdin = None
        stdout = None
        stderr = None

        def close(self) -> None:
            return None

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.returncode = -9

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float = 0) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    fake_process = FakeProcess()  # type: ignore[assignment]
    spec = SemanticModelSpec("sentence_transformers", "fake-1", "local-test-model", "r1", 8)
    provider._process = fake_process  # type: ignore[assignment]
    provider._spec = spec
    provider._send = lambda _payload: None  # type: ignore[assignment]

    def protocol_fault(**_kwargs):
        raise SemanticMemoryError("provider_protocol_invalid")

    provider._receive = protocol_fault  # type: ignore[assignment]
    with pytest.raises(SemanticMemoryError, match="provider_protocol_invalid"):
        provider.embed_documents(["文本"])
    assert provider._process is None

    # 非协议异常不触发终止，进程引用保留
    provider._process = fake_process  # type: ignore[assignment]
    provider._spec = spec

    def cancelled_fault(**_kwargs):
        raise concurrent.futures.CancelledError()

    provider._receive = cancelled_fault  # type: ignore[assignment]
    with pytest.raises(concurrent.futures.CancelledError):
        provider.embed_documents(["文本"])
    assert provider._process is fake_process

    # 轮询超时属协议类故障，仍走终止
    def timeout_fault(**_kwargs):
        raise SemanticMemoryError("timeout")

    provider._receive = timeout_fault  # type: ignore[assignment]
    with pytest.raises(SemanticMemoryError, match="timeout"):
        provider.embed_documents(["文本"])
    assert provider._process is None


def test_k_asr_cancel_retry_task_is_scheduled_once() -> None:
    """ASR 取消后自动重启预热任务只调度一次。"""

    from services.asr import ASRService
    from services.asr.service import ASR_CANCEL_RETRY_SECONDS

    service = ASRService(
        (sys.executable, "-u", "-c", _READY_WORKER),
        enabled=False,
        backend="sensevoice",
        model_name="SenseVoiceSmall",
        device="cpu",
        language="auto",
        timeout_seconds=5.0,
        startup_timeout_seconds=5.0,
        max_audio_bytes=4096,
    )

    async def scenario() -> None:
        service._schedule_cancel_retry()
        first = service._cancel_retry_task
        assert first is not None and not first.done()
        assert ASR_CANCEL_RETRY_SECONDS >= 5.0
        service._schedule_cancel_retry()
        assert service._cancel_retry_task is first
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        await service.aclose()

    asyncio.run(scenario())
    assert service._closed is True
