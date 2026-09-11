from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from services.plugins import (
    PluginContext,
    PluginDescriptor,
    PluginManager,
    PluginSettings,
    PluginState,
)


class _Plugin:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    async def load(self, context: PluginContext) -> None:
        self.events.append(f"{self.name}:load:{context.generation}")

    async def start(self) -> None:
        self.events.append(f"{self.name}:start")

    async def stop(self) -> None:
        self.events.append(f"{self.name}:stop")

    async def unload(self) -> None:
        self.events.append(f"{self.name}:unload")

    async def health(self) -> str:
        self.events.append(f"{self.name}:health")
        return "ready"


def _descriptor(name: str, events: list[str], **kwargs: object) -> PluginDescriptor:
    return PluginDescriptor(
        name,
        lambda context: _Plugin(events, context.plugin_id),
        **kwargs,
    )


def test_plugin_manager_runs_full_lifecycle_with_generation_and_health() -> None:
    events: list[str] = []
    manager = PluginManager()
    manager.register(_descriptor("alpha", events))

    async def scenario() -> None:
        started = await manager.start()
        assert started["running"] is True
        assert manager.status("alpha")["state"] == "running"
        generation = int(manager.status("alpha")["generation"])
        assert generation > 0

        health = await manager.health()
        assert health[0]["health"] == "ready"
        await manager.stop()
        assert manager.status("alpha")["state"] == "stopped"
        await manager.unload("alpha")
        assert manager.status("alpha")["state"] == "unloaded"
        assert events == [
            f"alpha:load:{generation}",
            "alpha:start",
            "alpha:health",
            "alpha:stop",
            "alpha:unload",
        ]
        await manager.close()

    asyncio.run(scenario())


def test_plugin_failure_isolated_and_public_status_hides_exception_text() -> None:
    events: list[str] = []

    def broken(_context: PluginContext) -> object:
        raise RuntimeError("secret-endpoint-token")

    manager = PluginManager()
    manager.register(PluginDescriptor("broken", broken))
    manager.register(_descriptor("healthy", events))

    async def scenario() -> None:
        await manager.start()
        broken_status = manager.status("broken")
        healthy_status = manager.status("healthy")
        assert broken_status["state"] == "failed"
        assert broken_status["error_type"] == "RuntimeError"
        assert healthy_status["state"] == "running"
        assert "secret-endpoint-token" not in repr(manager.status())
        await manager.close()

    asyncio.run(scenario())


def test_failed_file_plugin_is_retried_when_dependency_recovers(tmp_path: Path) -> None:
    directory = tmp_path / "plugins"
    directory.mkdir()
    gate = tmp_path / "ready"
    path = directory / "retry.py"
    path.write_text(
        "PLUGIN_ID = 'retry'\n"
        "def create_plugin(context):\n"
        "    from pathlib import Path\n"
        f"    if not Path({str(gate)!r}).exists():\n"
        "        raise RuntimeError('dependency unavailable')\n"
        "    class Plugin: pass\n"
        "    return Plugin()\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        settings=PluginSettings(directories=(str(directory),), reload_enabled=False)
    )

    async def scenario() -> None:
        await manager.start()
        assert manager.status("retry")["state"] == "failed"
        gate.write_text("ready", encoding="utf-8")
        await manager.refresh()
        assert manager.status("retry")["state"] == "running"
        await manager.close()

    asyncio.run(scenario())


def test_plugin_dependencies_load_in_order_and_failed_dependency_blocks_child() -> None:
    events: list[str] = []
    manager = PluginManager()
    manager.register(_descriptor("base", events))
    manager.register(_descriptor("child", events, dependencies=("base",)))

    async def scenario() -> None:
        await manager.start()
        assert events[:4] == ["base:load:1", "base:start", "child:load:2", "child:start"]
        await manager.close()

    asyncio.run(scenario())

    failed = PluginManager()

    def fail(_context: PluginContext) -> object:
        raise ValueError("failure")

    failed.register(PluginDescriptor("base", fail))
    failed.register(_descriptor("child", events, dependencies=("base",)))

    async def failed_scenario() -> None:
        await failed.start()
        assert failed.status("base")["state"] == "failed"
        assert failed.status("child")["state"] == "failed"
        assert failed.status("child")["error_type"] == "dependency_failed"
        await failed.close()

    asyncio.run(failed_scenario())


def test_plugin_directory_discovery_hot_reload_and_remove(tmp_path: Path) -> None:
    directory = tmp_path / "plugins"
    directory.mkdir()
    path = directory / "sample.py"
    path.write_text(
        "PLUGIN_ID = 'sample'\n"
        "PLUGIN_VERSION = '1'\n"
        "def create_plugin(context):\n"
        "    class Plugin:\n"
        "        async def start(self):\n"
        "            return None\n"
        "        async def unload(self):\n"
        "            return None\n"
        "        async def health(self):\n"
        "            return {'status': 'ready'}\n"
        "    return Plugin()\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        settings=PluginSettings(
            directories=(str(directory),),
            reload_interval_seconds=0.1,
        )
    )

    async def scenario() -> None:
        await manager.start()
        first = manager.status("sample")
        assert first["state"] == "running"
        first_generation = int(first["generation"])

        path.write_text(
            "PLUGIN_ID = 'sample'\n"
            "PLUGIN_VERSION = '2'\n"
            "def create_plugin(context):\n"
            "    class Plugin:\n"
            "        async def unload(self):\n"
            "            return None\n"
            "    return Plugin()\n",
            encoding="utf-8",
        )
        await manager.refresh()
        changed = manager.status("sample")
        assert changed["version"] == "2"
        assert int(changed["generation"]) > first_generation

        path.unlink()
        await manager.refresh()
        assert manager.status("sample")["status"] == "unavailable"
        await manager.close()

    asyncio.run(scenario())


def test_plugin_hot_reload_does_not_reuse_same_size_same_mtime_bytecode(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "plugins"
    directory.mkdir()
    path = directory / "stable.py"

    def source(version: str) -> str:
        return (
            "PLUGIN_ID = 'stable'\n"
            f"PLUGIN_VERSION = '{version}'\n"
            "def create_plugin(context):\n"
            "    class Plugin:\n"
            "        pass\n"
            "    return Plugin()\n"
        )

    first_source = source("1")
    second_source = source("2")
    assert len(first_source.encode("utf-8")) == len(second_source.encode("utf-8"))
    path.write_text(first_source, encoding="utf-8")
    original_stat = path.stat()
    manager = PluginManager(
        settings=PluginSettings(directories=(str(directory),), reload_enabled=False)
    )

    async def scenario() -> None:
        await manager.start()
        first = manager.status("stable")
        first_generation = int(first["generation"])
        assert first["version"] == "1"

        path.write_text(second_source, encoding="utf-8")
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        await manager.refresh()
        changed = manager.status("stable")
        assert changed["version"] == "2"
        assert int(changed["generation"]) > first_generation
        await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("second_id", ["stable", "renamed"])
def test_plugin_source_modules_survive_until_their_instances_unload(
    tmp_path: Path, second_id: str
) -> None:
    path = tmp_path / "stable.py"

    def source(plugin_id: str, version: str) -> str:
        return (
            "import sys\n"
            f"PLUGIN_ID = {plugin_id!r}\n"
            f"PLUGIN_VERSION = {version!r}\n"
            "def create_plugin(context):\n"
            "    assert sys.modules[__name__].PLUGIN_VERSION == PLUGIN_VERSION\n"
            "    class Plugin:\n"
            "        async def unload(self):\n"
            "            assert sys.modules[__name__].PLUGIN_VERSION == PLUGIN_VERSION\n"
            "    return Plugin()\n"
        )

    path.write_text(source("stable", "1"), encoding="utf-8")
    manager = PluginManager(
        settings=PluginSettings(directories=(str(tmp_path),), reload_enabled=False)
    )

    async def scenario() -> None:
        await manager.start()
        old_module = manager.descriptors()[0].factory.__module__
        try:
            assert old_module in sys.modules
            path.write_text(source(second_id, "2"), encoding="utf-8")
            await manager.refresh()
            status = manager.status(second_id)
            assert status["state"] == "running"
            assert status["version"] == "2"
            assert status["failure_count"] == 0
            assert manager.status()["discovery_error_count"] == 0
            assert old_module not in sys.modules
            new_module = manager.descriptors()[0].factory.__module__
            assert new_module in sys.modules
        finally:
            await manager.close()
        assert manager.status(second_id)["state"] == "unloaded"
        assert new_module not in sys.modules

    asyncio.run(scenario())


def test_plugin_reload_blocks_old_generation_service_calls_and_callbacks() -> None:
    class SharedService:
        def __init__(self) -> None:
            self.values: list[str] = []
            self.callbacks: list[object] = []

        def append(self, value: str) -> None:
            self.values.append(value)

        def subscribe(self, callback: object) -> None:
            self.callbacks.append(callback)

    shared = SharedService()
    contexts: list[PluginContext] = []

    class Plugin:
        def __init__(self, context: PluginContext) -> None:
            self.context = context

        async def start(self) -> None:
            service = self.context.services["shared"]
            service.subscribe(lambda: service.append(f"callback-{self.context.generation}"))

    def factory(context: PluginContext) -> Plugin:
        contexts.append(context)
        return Plugin(context)

    manager = PluginManager(services={"shared": shared})
    first_descriptor = PluginDescriptor("guarded", factory, fingerprint="one")
    second_descriptor = PluginDescriptor("guarded", factory, fingerprint="two")
    manager.register(first_descriptor)

    async def scenario() -> None:
        await manager.start()
        old_service = contexts[0].services["shared"]
        old_callback = shared.callbacks[0]

        await manager.reload("guarded", second_descriptor)
        assert len(contexts) == 2
        assert manager.status("guarded")["state"] == "running"

        # 回调注册表可能由共享服务持有；旧代次回调必须变为空操作。
        old_callback()
        assert shared.values == []
        with pytest.raises(RuntimeError, match="generation is inactive"):
            old_service.append("late-write")

        shared.callbacks[-1]()
        assert shared.values == [f"callback-{contexts[1].generation}"]
        await manager.close()

    asyncio.run(scenario())


def test_plugin_context_configuration_and_reload_hook_receive_new_snapshot() -> None:
    snapshots: list[tuple[int, object]] = []

    class Configurable:
        def __init__(self, context: PluginContext) -> None:
            self.context = context

        async def reload(self, context: PluginContext) -> None:
            snapshots.append((context.generation, context.configuration.get("mode")))

    manager = PluginManager(configuration={"mode": "one"})
    manager.register(PluginDescriptor("configurable", lambda context: Configurable(context)))

    async def scenario() -> None:
        await manager.start()
        await manager.reconfigure({"mode": "two"})
        assert snapshots and snapshots[-1][1] == "two"
        assert manager.status("configurable")["state"] == "running"
        await manager.close()

    asyncio.run(scenario())


def test_plugin_settings_reject_invalid_directory_and_interval() -> None:
    with pytest.raises(ValueError, match="plugins.directories"):
        PluginSettings.from_mapping({"directories": "plugins"})
    with pytest.raises(ValueError, match="reload_interval_seconds"):
        PluginSettings.from_mapping({"reload_interval_seconds": 0.01})


def test_plugin_manager_disabled_state_does_not_import_or_run_plugins() -> None:
    called = False

    def factory(_context: PluginContext) -> object:
        nonlocal called
        called = True
        return object()

    manager = PluginManager(settings=PluginSettings(enabled=False))
    manager.register(PluginDescriptor("disabled", factory))

    async def scenario() -> None:
        await manager.start()
        assert manager.status("disabled")["state"] == "disabled"
        assert called is False
        await manager.close()

    asyncio.run(scenario())


def test_plugin_reconfigure_disabled_unloads_instances_and_reenable_creates_new_generation() -> (
    None
):
    events: list[str] = []
    manager = PluginManager()
    manager.register(_descriptor("toggle", events))

    async def scenario() -> None:
        await manager.start()
        first_generation = int(manager.status("toggle")["generation"])
        disabled = await manager.reconfigure({"enabled": False})
        assert disabled["running"] is False
        assert manager.status("toggle")["state"] == "disabled"
        assert events[-2:] == ["toggle:stop", "toggle:unload"]

        await manager.reconfigure({"enabled": True})
        assert manager.status("toggle")["state"] == "running"
        assert int(manager.status("toggle")["generation"]) > first_generation
        await manager.close()

    asyncio.run(scenario())


def test_plugin_async_load_failure_awaits_unload_cleanup() -> None:
    events: list[str] = []

    class Broken:
        async def load(self, _context: PluginContext) -> None:
            events.append("load")
            raise RuntimeError("boom")

        async def unload(self) -> None:
            events.append("unload")

    manager = PluginManager()
    manager.register(PluginDescriptor("broken-cleanup", lambda _context: Broken()))

    async def scenario() -> None:
        await manager.start()
        assert events == ["load", "unload"]
        assert manager.status("broken-cleanup")["state"] == "failed"
        await manager.close()

    asyncio.run(scenario())


def test_plugin_configuration_change_without_reload_hook_recreates_instance() -> None:
    generations: list[int] = []
    unloads: list[int] = []

    class Configurable:
        def __init__(self, context: PluginContext) -> None:
            self.generation = context.generation
            generations.append(self.generation)

        async def unload(self) -> None:
            unloads.append(self.generation)

    manager = PluginManager(configuration={"mode": "one"})
    manager.register(PluginDescriptor("recreate", lambda context: Configurable(context)))

    async def scenario() -> None:
        await manager.start()
        first_generation = generations[-1]
        await manager.reconfigure({"mode": "two"})
        assert len(generations) == 2
        assert generations[-1] > first_generation
        assert unloads == [first_generation]
        assert manager.status("recreate")["state"] == "running"
        await manager.close()

    asyncio.run(scenario())


def test_plugin_initially_disabled_can_be_enabled_after_start() -> None:
    events: list[str] = []
    manager = PluginManager(settings=PluginSettings(enabled=False))
    manager.register(_descriptor("late-enable", events))

    async def scenario() -> None:
        await manager.start()
        assert manager.status("late-enable")["state"] == "disabled"
        await manager.reconfigure({"enabled": True})
        assert manager.status("late-enable")["state"] == "running"
        assert "late-enable:start" in events
        await manager.close()

    asyncio.run(scenario())


def test_plugin_reload_hook_failure_keeps_old_instance_and_gate() -> None:
    calls: list[str] = []
    instances: list[object] = []

    class Service:
        def __init__(self) -> None:
            self.values: list[str] = []

        def append(self, value: str) -> None:
            self.values.append(value)

    service = Service()

    class Plugin:
        def __init__(self, context: PluginContext) -> None:
            self.context = context

        async def reload(self, _context: PluginContext) -> None:
            calls.append("reload")
            raise RuntimeError("reload failed")

        async def unload(self) -> None:
            calls.append("unload")

    def factory(context: PluginContext) -> Plugin:
        plugin = Plugin(context)
        instances.append(plugin)
        return plugin

    manager = PluginManager(configuration={"mode": "one"}, services={"service": service})
    manager.register(PluginDescriptor("reload-failure", factory))

    async def scenario() -> None:
        await manager.start()
        first = instances[0]
        first_generation = int(manager.status("reload-failure")["generation"])
        await manager.reconfigure({"mode": "two"})
        assert calls == ["reload"]
        assert len(instances) == 1
        status = manager.status("reload-failure")
        assert status["state"] == "running"
        assert int(status["generation"]) == first_generation
        # 旧上下文仍然有效；配置失败没有替换实例或其服务代次。
        first.context.services["service"].append("ok")
        assert service.values == ["ok"]
        await manager.close()
        assert calls[-1] == "unload"

    asyncio.run(scenario())


def test_plugin_refresh_starts_new_file_plugin_when_manager_is_running(tmp_path: Path) -> None:
    directory = tmp_path / "plugins"
    directory.mkdir()
    manager = PluginManager(
        settings=PluginSettings(directories=(str(directory),), reload_enabled=False)
    )

    async def scenario() -> None:
        await manager.start()
        path = directory / "new.py"
        path.write_text(
            "PLUGIN_ID = 'new-file'\n"
            "def create_plugin(context):\n"
            "    class Plugin:\n"
            "        async def start(self):\n"
            "            return None\n"
            "    return Plugin()\n",
            encoding="utf-8",
        )
        await manager.refresh()
        assert manager.status("new-file")["state"] == "running"
        await manager.close()

    asyncio.run(scenario())


def test_plugin_discovery_error_status_does_not_expose_local_path(tmp_path: Path) -> None:
    directory = tmp_path / "plugins"
    directory.mkdir()
    path = directory / "broken.py"
    path.write_text("this is not valid python", encoding="utf-8")
    manager = PluginManager(
        settings=PluginSettings(directories=(str(directory),), reload_enabled=False)
    )

    async def scenario() -> None:
        await manager.start()
        status = manager.status()
        discovery_errors = status["discovery_errors"]
        assert discovery_errors
        rendered = repr(discovery_errors)
        assert str(tmp_path) not in rendered
        assert "broken.py" not in rendered
        await manager.close()

    asyncio.run(scenario())


def test_plugin_descriptor_rejects_duplicate_or_self_dependencies() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        PluginDescriptor("x", lambda _context: object(), dependencies=("y", "y"))
    with pytest.raises(ValueError, match="dependency"):
        PluginDescriptor("x", lambda _context: object(), dependencies=("x",))
    assert PluginState.RUNNING.value == "running"
