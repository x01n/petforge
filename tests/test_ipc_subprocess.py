from __future__ import annotations

import asyncio
import json
import logging
import sys
from time import monotonic

import pytest

from core.ipc import IPCProtocolSpec
from services.ipc import IPCProcessError, JSONLSubprocessClient
from services.ipc import subprocess as ipc_subprocess

_SPEC = IPCProtocolSpec(
    "meapet.test.jsonl",
    1,
    frozenset({"health", "wait", "cancel", "shutdown"}),
    frozenset({"ready", "health", "cancelled", "shutdown", "error"}),
    ("health", "shutdown"),
)


def _worker() -> str:
    return """
import json
import sys
import threading
import time

protocol = "meapet.test.jsonl"
version = 1
output_lock = threading.Lock()
cancel_event = threading.Event()
active_id = ""
def emit(value):
    with output_lock:
        print(json.dumps(value), flush=True)
def wait_for_cancel(request_id):
    if cancel_event.wait(10):
        emit({"type":"cancelled","protocol":protocol,"version":version,"request_id":request_id})
emit({"type":"ready","protocol":protocol,"version":version})
for line in sys.stdin:
    value = json.loads(line)
    event = {"protocol":protocol,"version":version,"request_id":value.get("request_id", "")}
    if value["type"] == "health":
        emit({**event,"type":"health","available":True})
    elif value["type"] == "wait":
        active_id = value.get("request_id", "")
        cancel_event.clear()
        threading.Thread(target=wait_for_cancel,args=(active_id,),daemon=True).start()
    elif value["type"] == "cancel" and value.get("request_id", "") == active_id:
        cancel_event.set()
    elif value["type"] == "shutdown":
        cancel_event.set()
        emit({**event,"type":"shutdown"})
        break
"""


def test_jsonl_subprocess_client_start_request_and_shutdown() -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", _worker()),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=1.0,
        backend="test-worker",
    )

    async def scenario() -> None:
        ready = await client.start()
        assert ready.type == "ready"
        process = client.process
        assert process is not None
        health = await client.request(
            "health",
            request_id="health-1",
            expected_events=frozenset({"health"}),
        )
        assert health.payload["available"] is True
        assert client.process is process
        await client.shutdown()
        assert client.process is None

    asyncio.run(scenario())


def test_jsonl_subprocess_timeout_recycles_worker_without_blocking_caller() -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", _worker()),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=0.05,
        shutdown_timeout_seconds=0.2,
        backend="test-timeout",
    )

    async def scenario() -> None:
        with pytest.raises(IPCProcessError) as captured:
            await client.request(
                "wait",
                request_id="wait-1",
                expected_events=frozenset({"health"}),
            )
        assert captured.value.reason_code == "response_timeout"
        assert client.process is None
        await client.shutdown()

    asyncio.run(scenario())


def test_jsonl_subprocess_interrupt_writes_while_request_waits() -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", _worker()),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=1.0,
        backend="test-interrupt",
    )

    async def scenario() -> None:
        request = asyncio.create_task(
            client.request(
                "wait",
                request_id="interrupt-1",
                expected_events=frozenset({"cancelled"}),
            )
        )
        await asyncio.sleep(0.05)
        await client.send_interrupt("cancel", request_id="interrupt-1")
        result = await request
        assert result.type == "cancelled"
        assert client.running is True
        await client.shutdown()

    asyncio.run(scenario())


def test_jsonl_subprocess_contract_does_not_log_payload_text(caplog) -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", _worker()),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=1.0,
        backend="test-private",
    )

    async def scenario() -> None:
        await client.request(
            "health",
            request_id="private-request-id",
            fields={"payload": "private speech text"},
            expected_events=frozenset({"health"}),
        )
        await client.shutdown()

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "ipc.request" in rendered
    assert "private speech text" not in rendered
    assert "private-request-id" not in rendered
    for record in caplog.records:
        json.loads(record.getMessage())


def test_jsonl_subprocess_rejects_invalid_business_contract_before_start() -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", _worker()),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=1.0,
        backend="test-contract",
    )

    async def scenario() -> None:
        invalid_calls = (
            client.request(
                "missing",
                request_id="request-1",
                expected_events=frozenset({"health"}),
            ),
            client.request(
                "health",
                request_id="  ",
                expected_events=frozenset({"health"}),
            ),
            client.request(
                "health",
                request_id="request-2",
                expected_events=frozenset(),
            ),
            client.request(
                "health",
                request_id="request-3",
                expected_events=frozenset({"ready", "unknown"}),
            ),
            client.request(
                "health",
                request_id="request-4",
                expected_events=frozenset({"health"}),
                passthrough_events=frozenset({"unknown"}),
            ),
        )
        for call in invalid_calls:
            with pytest.raises(ValueError):
                await call
        assert client.process is None

    asyncio.run(scenario())


def test_jsonl_subprocess_business_terminal_requires_exact_request_id() -> None:
    worker = _worker().replace(
        'emit({**event,"type":"health","available":True})',
        'emit({"type":"health","protocol":protocol,"version":version,"available":True})',
    )
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", worker),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.2,
        backend="test-terminal-id",
    )

    async def scenario() -> None:
        with pytest.raises(IPCProcessError) as captured:
            await client.request(
                "health",
                request_id="required-id",
                expected_events=frozenset({"health"}),
            )
        assert captured.value.reason_code == "request_id_mismatch"
        assert client.process is None
        await client.shutdown()

    asyncio.run(scenario())


def test_jsonl_subprocess_cancelling_queued_request_preserves_active_worker() -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", _worker()),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.2,
        backend="test-queued-cancel",
    )

    async def scenario() -> None:
        active = asyncio.create_task(
            client.request(
                "wait",
                request_id="active-1",
                expected_events=frozenset({"cancelled"}),
            )
        )
        for _index in range(100):
            submission = client._submissions.get("active-1")
            if submission is not None and submission.is_set():
                break
            await asyncio.sleep(0.005)
        process = client.process
        assert process is not None

        queued = asyncio.create_task(
            client.request(
                "health",
                request_id="queued-1",
                expected_events=frozenset({"health"}),
            )
        )
        for _index in range(100):
            submission = client._submissions.get("queued-1")
            if submission is not None:
                assert submission.is_set() is False
                break
            await asyncio.sleep(0.005)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert client.process is process
        assert client.running is True

        await client.send_interrupt("cancel", request_id="active-1")
        assert (await active).type == "cancelled"
        health = await client.request(
            "health",
            request_id="health-after-queue-cancel",
            expected_events=frozenset({"health"}),
        )
        assert health.payload["available"] is True
        assert client.process is process
        await client.shutdown()

    asyncio.run(scenario())


def test_jsonl_subprocess_cancelling_submitted_request_recycles_worker() -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", _worker()),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=5.0,
        shutdown_timeout_seconds=0.2,
        backend="test-submitted-cancel",
    )

    async def scenario() -> None:
        request = asyncio.create_task(
            client.request(
                "wait",
                request_id="submitted-1",
                expected_events=frozenset({"cancelled"}),
            )
        )
        for _index in range(100):
            submission = client._submissions.get("submitted-1")
            if submission is not None and submission.is_set():
                break
            await asyncio.sleep(0.005)
        first_process = client.process
        assert first_process is not None
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert client.process is None

        health = await client.request(
            "health",
            request_id="health-after-submit-cancel",
            expected_events=frozenset({"health"}),
        )
        assert health.payload["available"] is True
        assert client.process is not first_process
        await client.shutdown()

    asyncio.run(scenario())


def test_jsonl_subprocess_interrupted_dispose_never_reuses_poisoned_worker(monkeypatch) -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", _worker()),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=5.0,
        shutdown_timeout_seconds=0.2,
        backend="test-poisoned-worker",
    )
    dispose_entered = asyncio.Event()
    dispose_release = asyncio.Event()
    original_terminate = ipc_subprocess.terminate_process_tree

    async def blocked_terminate(*args, **kwargs) -> None:
        dispose_entered.set()
        await dispose_release.wait()
        await original_terminate(*args, **kwargs)

    monkeypatch.setattr(ipc_subprocess, "terminate_process_tree", blocked_terminate)

    async def scenario() -> None:
        request = asyncio.create_task(
            client.request(
                "wait",
                request_id="poisoned-1",
                expected_events=frozenset({"cancelled"}),
            )
        )
        for _index in range(100):
            submission = client._submissions.get("poisoned-1")
            if submission is not None and submission.is_set():
                break
            await asyncio.sleep(0.005)
        poisoned_process = client.process
        assert poisoned_process is not None

        request.cancel()
        await dispose_entered.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert client.process is poisoned_process
        assert client.running is False

        monkeypatch.setattr(ipc_subprocess, "terminate_process_tree", original_terminate)
        dispose_release.set()
        health = await client.request(
            "health",
            request_id="health-after-poison",
            expected_events=frozenset({"health"}),
        )
        assert health.payload["available"] is True
        assert client.process is not poisoned_process
        await client.shutdown()

    asyncio.run(scenario())


def test_jsonl_subprocess_shutdown_bypasses_long_request_and_is_bounded() -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", _worker()),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=30.0,
        shutdown_timeout_seconds=0.1,
        backend="test-bounded-shutdown",
    )

    async def scenario() -> None:
        request = asyncio.create_task(
            client.request(
                "wait",
                request_id="long-request",
                expected_events=frozenset({"cancelled"}),
            )
        )
        for _index in range(100):
            submission = client._submissions.get("long-request")
            if submission is not None and submission.is_set():
                break
            await asyncio.sleep(0.005)
        started = monotonic()
        await client.shutdown()
        assert monotonic() - started < 1.0
        assert client.process is None
        with pytest.raises(IPCProcessError):
            await request

    asyncio.run(scenario())


def test_jsonl_subprocess_cancelled_shutdown_keeps_cleanup_and_second_call_compensates() -> None:
    worker = _worker().replace(
        'cancel_event.set()\n        emit({**event,"type":"shutdown"})\n        break',
        "cancel_event.set()\n        time.sleep(10)",
    )
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", worker),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.1,
        backend="test-cancelled-shutdown",
    )

    async def scenario() -> None:
        await client.start()
        shutdown = asyncio.create_task(client.shutdown())
        await asyncio.sleep(0.02)
        shutdown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        for _index in range(100):
            if client.process is None:
                break
            await asyncio.sleep(0.01)
        assert client.process is None
        await asyncio.wait_for(client.shutdown(), timeout=0.2)

    asyncio.run(scenario())


def test_jsonl_subprocess_second_shutdown_retries_interrupted_internal_cleanup() -> None:
    worker = _worker().replace(
        'cancel_event.set()\n        emit({**event,"type":"shutdown"})\n        break',
        "cancel_event.set()\n        time.sleep(10)",
    )
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", worker),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.1,
        backend="test-shutdown-retry",
    )

    async def scenario() -> None:
        await client.start()
        first_waiter = asyncio.create_task(client.shutdown())
        for _index in range(100):
            cleanup = client._shutdown_task
            if cleanup is not None and not cleanup.done():
                break
            await asyncio.sleep(0.005)
        cleanup = client._shutdown_task
        assert cleanup is not None
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter
        assert client.process is not None

        await asyncio.wait_for(client.shutdown(), timeout=1.0)
        assert client.process is None

    asyncio.run(scenario())


def test_jsonl_subprocess_logs_shutdown_latency_and_cancelled_terminal(caplog) -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", _worker()),
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.2,
        backend="test-terminal-log",
    )

    async def scenario() -> None:
        request = asyncio.create_task(
            client.request(
                "wait",
                request_id="cancel-log-request",
                expected_events=frozenset({"cancelled"}),
            )
        )
        await asyncio.sleep(0.05)
        await client.send_interrupt("cancel", request_id="cancel-log-request")
        assert (await request).type == "cancelled"
        await client.shutdown()

    with caplog.at_level(logging.INFO, logger="services.ipc.subprocess"):
        asyncio.run(scenario())
    payloads = [json.loads(record.getMessage()) for record in caplog.records]
    request_terminals = [
        item
        for item in payloads
        if item.get("event") == "ipc.request" and item.get("status") != "started"
    ]
    assert [item["status"] for item in request_terminals] == ["cancelled"]
    shutdown_events = [item for item in payloads if item.get("event") == "ipc.worker.shutdown"]
    assert [item["status"] for item in shutdown_events] == ["started", "completed"]
    assert shutdown_events[-1]["duration_ms"] >= 0


def test_jsonl_subprocess_cancelled_start_is_info_event(caplog) -> None:
    client = JSONLSubprocessClient(
        _SPEC,
        (sys.executable, "-u", "-c", "import time; time.sleep(10)"),
        startup_timeout_seconds=2.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.2,
        backend="test-start-cancel-log",
    )

    async def scenario() -> None:
        start_task = asyncio.create_task(client.start())
        for _index in range(50):
            if client.process is not None:
                break
            await asyncio.sleep(0.01)
        assert client.process is not None
        start_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start_task
        await client.shutdown()

    with caplog.at_level(logging.INFO, logger="services.ipc.subprocess"):
        asyncio.run(scenario())
    records = [
        record
        for record in caplog.records
        if '"event": "ipc.worker.start"' in record.getMessage()
        and '"status": "cancelled"' in record.getMessage()
    ]
    assert len(records) == 1
    payload = json.loads(records[0].getMessage())
    assert payload["reason_code"] == "startup_cancelled"
    assert records[0].levelno == logging.INFO
