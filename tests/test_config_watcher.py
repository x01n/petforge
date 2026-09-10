from __future__ import annotations

import asyncio
from pathlib import Path

from config.loader import LoadedConfiguration, default_configuration_values, load_configuration
from config.watcher import ConfigurationWatcher


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _defaults(tmp_path: Path) -> dict[str, object]:
    return default_configuration_values(
        resource_root=tmp_path / "resources", database_path=tmp_path / "data.sqlite3"
    )


def test_watcher_ignores_initial_content_and_same_content_rewrites(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write(path, "app:\n  name: first\n")
    seen: list[str] = []

    async def on_reload(configuration: LoadedConfiguration) -> dict[str, str]:
        seen.append(str(configuration.values["app"]["name"]))
        return {"status": "reloaded"}

    watcher = ConfigurationWatcher(
        path,
        on_reload,
        defaults=_defaults(tmp_path),
        environment={},
        debounce_seconds=0,
        stable_checks=2,
    )

    async def scenario() -> None:
        started = await watcher.start()
        assert started["status"] == "running"
        _write(path, "app:\n  name: first\n")
        await watcher.poll_once()
        await watcher.poll_once()
        assert seen == []
        _write(path, "app:\n  name: second\n")
        assert (await watcher.poll_once())["status"] == "pending"
        assert (await watcher.poll_once())["status"] == "reloaded"
        assert seen == ["second"]
        await watcher.stop()

    asyncio.run(scenario())


def test_watcher_rejects_invalid_yaml_and_keeps_last_configuration(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write(path, "app:\n  name: stable\n")
    seen: list[str] = []

    def on_reload(configuration: LoadedConfiguration) -> dict[str, str]:
        seen.append(str(configuration.values["app"]["name"]))
        return {"status": "applied"}

    watcher = ConfigurationWatcher(
        path,
        on_reload,
        defaults=_defaults(tmp_path),
        environment={},
        debounce_seconds=0,
        stable_checks=1,
    )

    async def scenario() -> None:
        await watcher.start()
        _write(path, "app: [broken\n")
        result = await watcher.poll_once()
        assert result["status"] == "rejected"
        assert seen == []
        assert watcher.current_configuration is not None
        assert watcher.current_configuration.values["app"]["name"] == "stable"
        # 同一坏内容只尝试一次，不会每轮重复回调。
        assert (await watcher.poll_once())["status"] == "rejected"
        assert seen == []
        _write(path, "app:\n  name: recovered\n")
        assert (await watcher.poll_once())["status"] == "reloaded"
        assert seen == ["recovered"]
        assert watcher.generation == 1
        await watcher.stop()

    asyncio.run(scenario())


def test_watcher_start_preserves_initial_unavailable_status_until_file_appears(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.yaml"
    watcher = ConfigurationWatcher(
        path,
        lambda _configuration: {"status": "reloaded"},
        defaults=_defaults(tmp_path),
        environment={},
        debounce_seconds=0,
        stable_checks=1,
    )

    async def scenario() -> None:
        started = await watcher.start()
        assert started["status"] == "unavailable"
        assert started["running"] is True
        assert watcher.current_configuration is None

        _write(path, "app:\n  name: recovered\n")
        result = await watcher.poll_once()
        assert result["status"] == "reloaded"
        assert watcher.current_configuration is not None
        await watcher.stop()

    asyncio.run(scenario())


def test_watcher_start_preserves_initial_rejected_status_for_invalid_yaml(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.yaml"
    _write(path, "app: [broken\n")
    watcher = ConfigurationWatcher(
        path,
        lambda _configuration: {"status": "reloaded"},
        defaults=_defaults(tmp_path),
        environment={},
        debounce_seconds=0,
        stable_checks=1,
    )

    async def scenario() -> None:
        started = await watcher.start()
        assert started["status"] == "rejected"
        assert started["running"] is True
        assert watcher.current_configuration is None

        _write(path, "app:\n  name: recovered\n")
        result = await watcher.poll_once()
        assert result["status"] == "reloaded"
        assert watcher.current_configuration is not None
        await watcher.stop()

    asyncio.run(scenario())


def test_watcher_serializes_concurrent_poll_and_reload_callback(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write(path, "app:\n  name: first\n")
    callbacks = 0
    active = 0
    maximum_active = 0

    async def on_reload(_configuration: LoadedConfiguration) -> dict[str, str]:
        nonlocal callbacks, active, maximum_active
        callbacks += 1
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"status": "ok"}

    watcher = ConfigurationWatcher(
        path,
        on_reload,
        defaults=_defaults(tmp_path),
        environment={},
        debounce_seconds=0,
        stable_checks=1,
    )

    async def scenario() -> None:
        await watcher.start()
        _write(path, "app:\n  name: second\n")
        results = await asyncio.gather(*(watcher.poll_once() for _ in range(8)))
        assert callbacks == 1
        assert maximum_active == 1
        assert watcher.generation == 1
        assert any(result["status"] == "reloaded" for result in results)
        await watcher.stop()

    asyncio.run(scenario())


def test_watcher_debounces_unstable_direct_writes(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write(path, "app:\n  name: first\n")
    seen: list[str] = []
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    def on_reload(configuration: LoadedConfiguration) -> None:
        seen.append(str(configuration.values["app"]["name"]))

    watcher = ConfigurationWatcher(
        path,
        on_reload,
        defaults=_defaults(tmp_path),
        environment={},
        debounce_seconds=1,
        stable_checks=2,
        clock=clock,
    )

    async def scenario() -> None:
        nonlocal clock_value
        await watcher.start()
        _write(path, "app:\n  name: partial\n")
        assert (await watcher.poll_once())["status"] == "pending"
        _write(path, "app:\n  name: final\n")
        assert (await watcher.poll_once())["status"] == "pending"
        clock_value = 1.1
        assert (await watcher.poll_once())["status"] == "reloaded"
        # 内容连续稳定且超过防抖时间后才提交。
        assert seen == ["final"]
        await watcher.stop()

    asyncio.run(scenario())


def test_watcher_stop_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write(path, "{}\n")
    watcher = ConfigurationWatcher(path, lambda _configuration: None)

    async def scenario() -> None:
        await watcher.start()
        first = await watcher.stop()
        second = await watcher.stop()
        assert first["status"] == second["status"] == "stopped"
        assert watcher.running is False

    asyncio.run(scenario())


def test_watcher_can_stop_itself_from_reload_callback(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write(path, "app:\n  name: first\n")
    watcher: ConfigurationWatcher

    async def on_reload(_configuration: LoadedConfiguration) -> dict[str, str]:
        await watcher.stop()
        return {"status": "reloaded"}

    watcher = ConfigurationWatcher(
        path,
        on_reload,
        defaults=_defaults(tmp_path),
        environment={},
        debounce_seconds=0,
        stable_checks=1,
    )

    async def scenario() -> None:
        await watcher.start()
        _write(path, "app:\n  name: second\n")
        result = await watcher.poll_once()
        assert result["status"] == "reloaded"
        await asyncio.sleep(0)
        assert watcher.running is False

    asyncio.run(scenario())


def test_watcher_retries_explicitly_deferred_reload(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write(path, "app:\n  name: first\n")
    calls = 0
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    def on_reload(_configuration: LoadedConfiguration) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status": "restart_required", "retry": calls == 1}

    watcher = ConfigurationWatcher(
        path,
        on_reload,
        defaults=_defaults(tmp_path),
        environment={},
        debounce_seconds=0,
        stable_checks=1,
        interval_seconds=1,
        clock=clock,
    )

    async def scenario() -> None:
        nonlocal clock_value
        await watcher.start()
        _write(path, "app:\n  name: second\n")
        assert (await watcher.poll_once())["status"] == "deferred"
        assert calls == 1
        assert (await watcher.poll_once())["status"] == "deferred"
        clock_value = 1.1
        assert (await watcher.poll_once())["status"] == "restart_required"
        assert calls == 2
        assert watcher.current_configuration is not None
        assert watcher.current_configuration.values["app"]["name"] == "second"
        await watcher.stop()

    asyncio.run(scenario())


def test_watcher_acknowledge_updates_baseline_without_duplicate_callback(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write(path, "app:\n  name: first\n")
    seen: list[str] = []

    def on_reload(configuration: LoadedConfiguration) -> None:
        seen.append(str(configuration.values["app"]["name"]))

    initial = LoadedConfiguration(path.resolve(), {"app": {"name": "first"}})
    updated = LoadedConfiguration(path.resolve(), {"app": {"name": "second"}})
    watcher = ConfigurationWatcher(
        path,
        on_reload,
        initial_configuration=initial,
        debounce_seconds=0,
        stable_checks=1,
    )

    async def scenario() -> None:
        await watcher.start()
        _write(path, "app:\n  name: second\n")
        assert watcher.acknowledge(updated) is True
        await asyncio.sleep(0)
        assert watcher.generation == 0
        assert seen == []
        assert (await watcher.poll_once())["status"] == "running"
        assert seen == []
        await watcher.stop()

    asyncio.run(scenario())


def test_watcher_acknowledge_does_not_hide_newer_external_write(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write(path, "app:\n  name: first\n")
    seen: list[str] = []

    def on_reload(configuration: LoadedConfiguration) -> None:
        seen.append(str(configuration.values["app"]["name"]))

    defaults = _defaults(tmp_path)
    initial = load_configuration(
        path,
        defaults=defaults,
        environment={},
    )
    watcher = ConfigurationWatcher(
        path,
        on_reload,
        defaults=defaults,
        environment={},
        initial_configuration=initial,
        debounce_seconds=0,
        stable_checks=1,
    )

    async def scenario() -> None:
        await watcher.start()
        _write(path, "app:\n  name: second\n")
        applied = load_configuration(
            path,
            defaults=defaults,
            environment={},
        )
        # 外部编辑器在运行时确认前又写入第三份内容；确认第二份不能吞掉第三份。
        _write(path, "app:\n  name: third\n")
        assert watcher.acknowledge(applied) is True
        result = await watcher.poll_once()
        assert result["status"] == "reloaded"
        assert seen == ["third"]
        assert watcher.current_configuration is not None
        assert watcher.current_configuration.values["app"]["name"] == "third"
        await watcher.stop()

    asyncio.run(scenario())
