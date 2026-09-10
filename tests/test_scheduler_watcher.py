import asyncio
import threading
import time

from services.scheduler import DesktopWindowWatcher, TriggerService


class FakePlatform:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)

    def foreground_window(self):
        return (
            self.snapshots.pop(0)
            if self.snapshots
            else {
                "status": "available",
                "backend": "x11",
                "window_id": "one",
                "app_id": "app",
                "pid": 1,
                "process_name": "app",
                "executable": "/app",
            }
        )


def test_watcher_emits_only_on_identity_change_and_supports_initial_snapshot():
    seen = []

    async def run(_action, _trigger, payload):
        seen.append(payload)

    first = {
        "status": "available",
        "backend": "x11",
        "window_id": "one",
        "app_id": "app",
        "pid": 1,
        "process_name": "app",
        "executable": "/app",
    }
    second = dict(first, window_id="two")
    triggers = TriggerService(action_runner=run)
    triggers.register(
        trigger_id="w",
        event_name="window_changed",
        action={"name": "noop"},
        owner="test",
    )
    watcher = DesktopWindowWatcher(
        platform=FakePlatform([first, first, second]), triggers=triggers, emit_initial=True
    )

    async def scenario():
        await watcher.poll_once()
        await watcher.poll_once()
        await watcher.poll_once()

    asyncio.run(scenario())
    assert [item["window_id"] for item in seen] == ["one", "two"]


def test_watcher_platform_failure_is_reported_without_task_failure():
    class Broken:
        def foreground_window(self):
            raise RuntimeError("desktop unavailable")

    watcher = DesktopWindowWatcher(platform=Broken(), triggers=TriggerService(), poll_seconds=0)
    asyncio.run(watcher.poll_once())
    assert watcher.status()["status"] == "unavailable"
    assert watcher.status()["error"] == "desktop unavailable"


def test_watcher_start_stop_is_cancellable():
    watcher = DesktopWindowWatcher(
        platform=FakePlatform([]), triggers=TriggerService(), poll_seconds=0.05
    )

    async def scenario():
        started = await watcher.start()
        assert started["status"] == "running"
        await asyncio.sleep(0.06)
        stopped = await watcher.stop()
        assert stopped["status"] == "stopped"

    asyncio.run(scenario())


def test_watcher_survives_trigger_action_failure():
    class BrokenTriggers(TriggerService):
        async def emit(self, _event_name, _payload=None):
            raise RuntimeError("trigger failed")

    watcher = DesktopWindowWatcher(
        platform=FakePlatform(
            [
                {
                    "status": "available",
                    "backend": "x11",
                    "window_id": "one",
                }
            ]
        ),
        triggers=BrokenTriggers(),
        poll_seconds=0.05,
        emit_initial=True,
    )

    async def scenario():
        await watcher.start()
        await asyncio.sleep(0.06)
        assert watcher.running
        await watcher.stop()

    asyncio.run(scenario())


def test_watcher_timeout_reuses_one_sync_read_and_recovers():
    started = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    class Blocking:
        def foreground_window(self):
            nonlocal calls
            with calls_lock:
                calls += 1
            started.set()
            release.wait(timeout=2)
            return {"status": "available", "backend": "x11", "window_id": "one"}

    watcher = DesktopWindowWatcher(
        platform=Blocking(),
        triggers=TriggerService(),
        poll_timeout_seconds=0.1,
        poll_seconds=0.05,
    )

    async def scenario():
        first = await watcher.poll_once()
        assert first["status"] == "degraded"
        assert first["foreground_window_read"] == {
            "status": "timed_out",
            "in_flight": True,
            "recoverable": True,
            "stale_in_flight": 0,
        }
        assert started.is_set()
        for _ in range(3):
            await watcher.poll_once()
        assert calls == 1
        release.set()
        deadline = asyncio.get_running_loop().time() + 1
        recovered = await watcher.poll_once()
        while recovered["status"] != "running":
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("同步读取线程未在释放后恢复")
            await asyncio.sleep(0.01)
            recovered = await watcher.poll_once()
        assert recovered["status"] == "running"
        assert calls == 1

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_watcher_stop_is_fast_while_sync_read_cannot_be_cancelled():
    started = threading.Event()
    release = threading.Event()

    class Blocking:
        def foreground_window(self):
            started.set()
            release.wait(timeout=2)
            return {"status": "available", "backend": "x11", "window_id": "one"}

    watcher = DesktopWindowWatcher(
        platform=Blocking(),
        triggers=TriggerService(),
        poll_seconds=0.05,
        poll_timeout_seconds=0.1,
    )

    async def scenario():
        await watcher.start()
        deadline = asyncio.get_running_loop().time() + 1
        while not started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("同步读取线程未启动")
            await asyncio.sleep(0.01)
        started_at = time.monotonic()
        stopped = await watcher.stop()
        assert time.monotonic() - started_at < 0.2
        assert stopped["status"] == "stopped"
        assert stopped["foreground_window_read"] == {
            "status": "stale_in_flight",
            "in_flight": True,
            "recoverable": True,
            "stale_in_flight": 1,
        }

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_watcher_restart_discards_stale_read_and_uses_new_platform():
    old_started = threading.Event()
    old_release = threading.Event()
    old_calls = 0
    new_calls = 0

    class Blocking:
        def foreground_window(self):
            nonlocal old_calls
            old_calls += 1
            old_started.set()
            old_release.wait(timeout=2)
            return {"status": "available", "backend": "x11", "window_id": "old"}

    class Fast:
        def foreground_window(self):
            nonlocal new_calls
            new_calls += 1
            return {"status": "available", "backend": "x11", "window_id": "new"}

    watcher = DesktopWindowWatcher(
        platform=Blocking(),
        triggers=TriggerService(),
        poll_seconds=0.05,
        poll_timeout_seconds=0.1,
    )

    async def scenario():
        await watcher.start()
        deadline = asyncio.get_running_loop().time() + 1
        while not old_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("旧代同步读取线程未启动")
            await asyncio.sleep(0.01)
        await watcher.stop()

        watcher.set_platform(Fast())
        await watcher.start()
        deadline = asyncio.get_running_loop().time() + 1
        while new_calls == 0:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("新代同步读取未执行")
            await asyncio.sleep(0.01)
        assert old_calls == 1
        assert watcher.status()["snapshot"]["window_id"] == "new"
        read_status = watcher.status()["foreground_window_read"]
        assert read_status["stale_in_flight"] == 1
        assert read_status["recoverable"] is True
        await watcher.stop()

    try:
        asyncio.run(scenario())
    finally:
        old_release.set()


def test_watcher_running_platform_switch_uses_new_platform_generation():
    old_started = threading.Event()
    old_release = threading.Event()
    new_calls = 0

    class Blocking:
        def foreground_window(self):
            old_started.set()
            old_release.wait(timeout=2)
            return {"status": "available", "backend": "x11", "window_id": "old"}

    class Fast:
        def foreground_window(self):
            nonlocal new_calls
            new_calls += 1
            return {"status": "available", "backend": "x11", "window_id": "new"}

    watcher = DesktopWindowWatcher(
        platform=Blocking(),
        triggers=TriggerService(),
        poll_seconds=0.05,
        poll_timeout_seconds=0.1,
    )

    async def scenario():
        await watcher.start()
        deadline = asyncio.get_running_loop().time() + 1
        while not old_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("旧平台同步读取线程未启动")
            await asyncio.sleep(0.01)
        watcher.set_platform(Fast())
        while new_calls == 0:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("运行中平台切换后新读取未执行")
            await asyncio.sleep(0.01)
        assert watcher.status()["snapshot"]["window_id"] == "new"
        await watcher.stop()

    try:
        asyncio.run(scenario())
    finally:
        old_release.set()


def test_watcher_running_platform_switch_caps_stale_reads():
    releases = [threading.Event() for _ in range(3)]
    started = [threading.Event() for _ in range(3)]
    calls = [0, 0, 0]

    def make_platform(index: int):
        class Blocking:
            def foreground_window(self):
                calls[index] += 1
                started[index].set()
                releases[index].wait(timeout=2)
                return {"status": "available", "backend": "x11", "window_id": str(index)}

        return Blocking()

    watcher = DesktopWindowWatcher(
        platform=make_platform(0),
        triggers=TriggerService(),
        poll_seconds=60,
        poll_timeout_seconds=0.1,
    )

    async def scenario():
        await watcher.start()
        deadline = asyncio.get_running_loop().time() + 1
        while not started[0].is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("初始平台同步读取未启动")
            await asyncio.sleep(0.01)
        watcher.set_platform(make_platform(1))
        await watcher.poll_once()
        assert started[1].is_set()
        watcher.set_platform(make_platform(2))
        await watcher.poll_once()
        await asyncio.sleep(0.15)
        assert calls == [1, 1, 0]
        assert watcher.status()["status"] == "degraded"
        assert watcher.status()["foreground_window_read"]["stale_in_flight"] == 2
        await watcher.stop()

    try:
        asyncio.run(scenario())
    finally:
        for release in releases:
            release.set()


def test_watcher_refuses_unbounded_restart_while_stale_read_is_blocked():
    started = threading.Event()
    release = threading.Event()
    calls = 0

    class Blocking:
        def foreground_window(self):
            nonlocal calls
            calls += 1
            started.set()
            release.wait(timeout=2)
            return {"status": "available", "window_id": "old"}

    watcher = DesktopWindowWatcher(
        platform=Blocking(),
        triggers=TriggerService(),
        poll_seconds=0.05,
        poll_timeout_seconds=0.1,
    )

    async def scenario():
        await watcher.start()
        deadline = asyncio.get_running_loop().time() + 1
        while not started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("同步读取线程未启动")
            await asyncio.sleep(0.01)
        await watcher.stop()
        await watcher.start()
        deadline = asyncio.get_running_loop().time() + 1
        while calls < 2:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("第二代同步读取线程未启动")
            await asyncio.sleep(0.01)
        await watcher.stop()
        blocked_restart = await watcher.start()
        assert blocked_restart["status"] == "degraded"
        assert not watcher.running

    try:
        asyncio.run(scenario())
    finally:
        release.set()
