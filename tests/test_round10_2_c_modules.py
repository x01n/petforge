"""第 10 轮方向 2：模块管理器热装载/卸载/禁用语义的纯内存路径覆盖补漏。

覆盖面基于对 ``src/services/modules/manager.py`` 的实读：

- Manager 不暴露 enable/disable/restart 方法；"禁用/启用切换"由
  ``register(descriptor, replace=True)`` 配合 ``enabled=False`` 描述符承载；
- 运行中模块的同步注册会被拒绝，必须走异步 reload 路径。

全部为纯内存假对象，无网络、无磁盘、无 GUI。
"""

from __future__ import annotations

import asyncio

import pytest

from services.modules import ModuleDescriptor, ModuleManager, ModuleState


class _FakeModule:
    """带事件记录的假模块；覆盖 load/start/stop/unload/health 框架钩子。"""

    def __init__(
        self,
        events: list[str],
        name: str,
        *,
        fail_load: bool = False,
        fail_start: bool = False,
    ) -> None:
        self.events = events
        self.name = name
        self.fail_load = fail_load
        self.fail_start = fail_start

    async def load(self, _context) -> None:
        if self.fail_load:
            raise RuntimeError("load-boom")
        self.events.append(self.name + ":load")

    async def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("start-boom")
        self.events.append(self.name + ":start")

    async def stop(self) -> None:
        self.events.append(self.name + ":stop")

    async def unload(self) -> None:
        self.events.append(self.name + ":unload")

    async def health(self) -> str:
        return "healthy"


def test_register_projects_enabled_flag_and_rejects_unreplaced_duplicates() -> None:
    manager = ModuleManager()
    manager.register(ModuleDescriptor("enabled-mod", lambda _ctx: _FakeModule([], "on")))
    manager.register(
        ModuleDescriptor("disabled-mod", lambda _ctx: _FakeModule([], "off"), enabled=False)
    )
    assert manager.status("enabled-mod")["state"] == ModuleState.REGISTERED.value
    assert manager.status("disabled-mod")["state"] == ModuleState.DISABLED.value
    with pytest.raises(ValueError, match="duplicate module id"):
        manager.register(ModuleDescriptor("enabled-mod", lambda _ctx: _FakeModule([], "on")))
    with pytest.raises(TypeError, match="descriptor is required"):
        manager.register(object())  # type: ignore[arg-type]


def test_running_module_register_replacement_must_be_asynchronous() -> None:
    async def scenario() -> None:
        manager = ModuleManager()
        manager.register(ModuleDescriptor("busy-mod", lambda _ctx: _FakeModule([], "busy")))
        await manager.start()
        assert manager.status("busy-mod")["state"] == ModuleState.RUNNING.value
        with pytest.raises(RuntimeError, match="reloaded asynchronously"):
            manager.register(
                ModuleDescriptor("busy-mod", lambda _ctx: _FakeModule([], "busy")),
                replace=True,
            )
        await manager.close()

    asyncio.run(scenario())


def test_disable_and_enable_via_descriptor_replacement() -> None:
    """卸载后换入 enabled=False 描述符即为禁用；再换回启用描述符可重启。"""

    async def scenario() -> None:
        events: list[str] = []
        manager = ModuleManager()
        manager.register(ModuleDescriptor("toggle-mod", lambda _ctx: _FakeModule(events, "t")))
        await manager.start()
        assert manager.status("toggle-mod")["state"] == ModuleState.RUNNING.value

        await manager.unload("toggle-mod")
        manager.register(
            ModuleDescriptor("toggle-mod", lambda _ctx: _FakeModule(events, "t"), enabled=False),
            replace=True,
        )
        assert manager.status("toggle-mod")["state"] == ModuleState.DISABLED.value

        manager.register(
            ModuleDescriptor("toggle-mod", lambda _ctx: _FakeModule(events, "t"), enabled=True),
            replace=True,
        )
        await manager.start()
        assert manager.status("toggle-mod")["state"] == ModuleState.RUNNING.value
        assert events.count("t:start") == 2
        await manager.close()

    asyncio.run(scenario())


def test_failed_dependency_keeps_child_down_with_safe_error_metadata() -> None:
    async def scenario() -> None:
        manager = ModuleManager()
        manager.register(
            ModuleDescriptor("broken-root", lambda _ctx: _FakeModule([], "root", fail_start=True))
        )
        manager.register(
            ModuleDescriptor(
                "dependent-leaf",
                lambda _ctx: _FakeModule([], "leaf"),
                dependencies=("broken-root",),
            )
        )

        await manager.start()
        assert manager.status("broken-root")["state"] == ModuleState.FAILED.value
        assert manager.status("broken-root")["error_type"] == "RuntimeError"
        assert manager.status("dependent-leaf")["state"] == ModuleState.FAILED.value
        assert manager.status("dependent-leaf")["error_type"] == "dependency_failed"
        payload = repr(manager.status())
        assert "start-boom" not in payload
        assert manager.running
        await manager.close()

    asyncio.run(scenario())


def test_reload_with_replacement_swaps_instance_under_running_state() -> None:
    events: list[str] = []

    async def scenario() -> None:
        manager = ModuleManager()
        manager.register(ModuleDescriptor("swap-mod", lambda _ctx: _FakeModule(events, "old")))
        await manager.start()
        first_generation = manager.status("swap-mod")["generation"]

        new_instance = _FakeModule(events, "new")
        result = await manager.reload("swap-mod", replacement=new_instance)

        assert result["reload_status"] == "reloaded"
        assert result["state"] == ModuleState.RUNNING.value
        assert manager.status("swap-mod")["generation"] == first_generation + 1

    asyncio.run(scenario())


def test_unload_and_close_are_idempotent_with_unknown_ids() -> None:
    events: list[str] = []

    async def scenario() -> None:
        manager = ModuleManager()
        manager.register(ModuleDescriptor("known-mod", lambda _ctx: _FakeModule(events, "known")))

        await manager.start()
        await manager.unload("never-registered")
        assert manager.status()["closed"] is False

        await manager.unload()
        assert manager.status("known-mod")["state"] == ModuleState.UNLOADED.value

        await manager.close()
        assert manager.status()["closed"] is True
        await manager.close()
        assert manager.status()["closed"] is True

    asyncio.run(scenario())
