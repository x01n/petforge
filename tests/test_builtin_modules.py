from __future__ import annotations

import asyncio

from services.modules import ModuleDescriptor, ModuleGenerationGate, ModuleManager, ModuleState


class _Module:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    async def load(self, _context) -> None:
        self.events.append(self.name + ":load")

    async def start(self) -> None:
        self.events.append(self.name + ":start")

    async def stop(self) -> None:
        self.events.append(self.name + ":stop")

    async def unload(self) -> None:
        self.events.append(self.name + ":unload")

    async def health(self) -> str:
        return "ready"


def test_adopted_modules_are_registered_without_duplicate_start() -> None:
    events: list[str] = []
    manager = ModuleManager()
    manager.adopt("tts", _Module(events, "tts"))

    async def scenario() -> None:
        await manager.start()
        assert manager.status("tts")["state"] == ModuleState.RUNNING.value
        assert events == []
        await manager.close()
        assert events == []

    asyncio.run(scenario())


def test_module_dependencies_and_failure_are_isolated() -> None:
    events: list[str] = []
    manager = ModuleManager()

    def broken(_context):
        raise ValueError("private endpoint")

    manager.register(ModuleDescriptor("broken", broken))
    manager.register(
        ModuleDescriptor(
            "child",
            lambda context: _Module(events, "child"),
            dependencies=("broken",),
        )
    )
    manager.register(ModuleDescriptor("healthy", lambda context: _Module(events, "healthy")))

    async def scenario() -> None:
        await manager.start()
        assert manager.status("broken")["state"] == ModuleState.FAILED.value
        assert manager.status("child")["error_type"] == "dependency_failed"
        assert manager.status("healthy")["state"] == ModuleState.RUNNING.value
        assert manager.status("broken")["error_type"] == "ValueError"
        assert "private endpoint" not in repr(manager.status())
        await manager.close()

    asyncio.run(scenario())


def test_reload_deactivates_old_generation_and_accepts_async_replacement() -> None:
    events: list[str] = []
    manager = ModuleManager()
    manager.register(ModuleDescriptor("model", lambda context: _Module(events, "old")))

    async def scenario() -> None:
        await manager.start()
        first_gate = manager._records["model"].gate
        assert first_gate is not None
        first_generation = int(manager.status("model")["generation"])
        await manager.reload(
            "model",
            replacement=lambda context: _Module(events, "new"),
        )
        assert not first_gate.active
        status = manager.status("model")
        assert status["state"] == ModuleState.RUNNING.value
        assert int(status["generation"]) > first_generation
        assert events == [
            "old:load",
            "old:start",
            "old:stop",
            "old:unload",
            "new:load",
            "new:start",
        ]
        await manager.close()

    asyncio.run(scenario())


def test_reload_restores_old_instance_when_new_generation_fails() -> None:
    events: list[str] = []
    manager = ModuleManager()
    manager.register(ModuleDescriptor("tts", lambda context: _Module(events, "old")))

    def broken(_context):
        raise RuntimeError("new generation failed")

    async def scenario() -> None:
        await manager.start()
        old_gate = manager._records["tts"].gate
        assert old_gate is not None
        before_generation = int(manager.status("tts")["generation"])

        result = await manager.reload("tts", replacement=broken)

        assert result["state"] == ModuleState.RUNNING.value
        assert int(result["generation"]) > before_generation
        assert not old_gate.active
        assert manager._records["tts"].instance is not None
        assert events == [
            "old:load",
            "old:start",
            "old:stop",
            "old:unload",
            "old:load",
            "old:start",
        ]
        await manager.close()

    asyncio.run(scenario())



def test_reconfigure_reports_restored_old_instance_as_failure() -> None:
    events: list[str] = []
    manager = ModuleManager()

    class Reloadable(_Module):
        async def reload(self, _context):
            return _Module(events, "new")

    class BrokenStart(_Module):
        async def start(self) -> None:
            raise RuntimeError("replacement start failed")

    class FactoryModule(_Module):
        async def reload(self, _context):
            return BrokenStart(events, "broken")

    manager.register(ModuleDescriptor("tts", lambda context: FactoryModule(events, "old")))

    async def scenario() -> None:
        await manager.start()
        result = await manager.reconfigure({"voice": "broken"})
        assert result["status"] == "degraded"
        assert result["reloaded"] == ()
        assert result["failed"] == ("tts",)
        status = manager.status("tts")
        assert status["state"] == ModuleState.RUNNING.value
        await manager.close()

    asyncio.run(scenario())


def test_reload_cancellation_restores_old_instance_and_cleans_new_generation() -> None:
    events: list[str] = []
    manager = ModuleManager()
    manager.register(ModuleDescriptor("tts", lambda context: _Module(events, "old")))

    async def scenario() -> None:
        await manager.start()
        started = asyncio.Event()
        release = asyncio.Event()

        class SlowModule(_Module):
            async def start(self) -> None:
                events.append("new:start")
                started.set()
                await release.wait()

        task = asyncio.create_task(
            manager.reload("tts", replacement=lambda context: SlowModule(events, "new"))
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        result = manager.status("tts")
        assert result["state"] == ModuleState.RUNNING.value
        assert isinstance(manager._records["tts"].instance, _Module)
        assert events == [
            "old:load",
            "old:start",
            "old:stop",
            "old:unload",
            "new:load",
            "new:start",
            "new:stop",
            "new:unload",
            "old:load",
            "old:start",
        ]
        await manager.close()

    asyncio.run(scenario())


def test_generation_gate_drops_late_callbacks() -> None:
    gate = ModuleGenerationGate(1)
    called: list[str] = []
    callback = gate.guard_callback(lambda: called.append("called"))
    callback()
    gate.deactivate()
    assert callback() is None
    assert called == ["called"]


def test_reconfigure_invokes_async_reload_hook_with_context() -> None:
    seen: list[tuple[str, int]] = []

    class Reloadable:
        async def reload(self, context) -> None:
            seen.append((context.configuration["value"], context.generation))

    manager = ModuleManager(configuration={"value": "one"})
    manager.register(ModuleDescriptor("memory", lambda context: Reloadable()))

    async def scenario() -> None:
        await manager.start()
        result = await manager.reconfigure({"value": "two"})
        assert result["reloaded"] == ("memory",)
        assert seen == [("two", 1)]
        await manager.close()

    asyncio.run(scenario())


def test_reconfigure_replaces_returned_instance_and_advances_generation() -> None:
    events: list[str] = []
    old_gate_seen: list[ModuleGenerationGate] = []

    class Reloadable(_Module):
        async def reload(self, _context):
            return _Module(events, "new")

    manager = ModuleManager()
    manager.register(ModuleDescriptor("tts", lambda context: Reloadable(events, "old")))

    async def scenario() -> None:
        await manager.start()
        gate = manager._records["tts"].gate
        assert gate is not None
        old_gate_seen.append(gate)
        before_generation = int(manager.status("tts")["generation"])

        result = await manager.reconfigure({"voice": "new"})

        assert result["status"] == "reloaded"
        assert result["reloaded"] == ("tts",)
        status = manager.status("tts")
        assert status["state"] == ModuleState.RUNNING.value
        assert int(status["generation"]) > before_generation
        assert not old_gate_seen[0].active
        assert events == [
            "old:load",
            "old:start",
            "old:stop",
            "old:unload",
            "new:load",
            "new:start",
        ]
        await manager.close()

    asyncio.run(scenario())
