from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest

from app.runtime import build_runtime
from config.loader import LoadedConfiguration
from config.resources import inspect_resources
from core.asr import ASRHealth
from core.tts.contracts import EngineHealth
from services.tools.types import ToolCallContext
from services.tts.coordinator import TTSCoordinator


def _configuration(tmp_path: Path, values: dict[str, object]) -> LoadedConfiguration:
    return LoadedConfiguration(tmp_path / "config.yaml", values)


def _runtime(tmp_path: Path, values: dict[str, object]):
    configuration = _configuration(tmp_path, values)
    return configuration, build_runtime(
        configuration,
        inspect_resources(tmp_path / "resources"),
    )


def test_reload_cancellation_during_new_tts_start_closes_replacement_and_resumes_old_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module

    initial, runtime = _runtime(
        tmp_path,
        {
            "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
            "asr": {"enabled": False},
            "scheduler": {"enabled": False, "activity": {"enabled": False}},
            "watcher": {"enabled": False},
            "behavior": {"enabled": False},
            "config": {"reload": {"enabled": False}},
        },
    )
    old_first_started = asyncio.Event()
    old_restarted = asyncio.Event()
    replacement_started = asyncio.Event()
    replacement_closed = asyncio.Event()

    class OldBackend:
        def __init__(self) -> None:
            self.start_calls = 0

        async def start(self) -> EngineHealth:
            self.start_calls += 1
            if self.start_calls == 1:
                old_first_started.set()
                await asyncio.Event().wait()
            old_restarted.set()
            return EngineHealth("old", True, "ready")

        async def health(self) -> EngineHealth:
            return EngineHealth("old", True, "ready")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            return None

    class ReplacementBackend:
        async def start(self) -> EngineHealth:
            replacement_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def health(self) -> EngineHealth:
            return EngineHealth("replacement", False, "loading")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            replacement_closed.set()

    old_backend = OldBackend()
    replacement = TTSCoordinator(ReplacementBackend())
    runtime.tts.backend = old_backend
    monkeypatch.setattr(runtime_module, "_build_tts", lambda _configuration: replacement)
    updated = _configuration(
        tmp_path,
        {**initial.values, "tts": {"enabled": False, "backend": "text_only", "language": "ja"}},
    )

    async def scenario() -> None:
        await runtime.start_background()
        await old_first_started.wait()
        task = asyncio.create_task(runtime.apply_configuration(updated))
        await replacement_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(replacement_closed.wait(), timeout=1.0)
        await asyncio.wait_for(old_restarted.wait(), timeout=1.0)
        assert runtime.configuration is initial
        assert runtime.tts.backend is old_backend
        assert old_backend.start_calls == 2
        assert not runtime._reload_cleanup_tasks
        await runtime.close()

    asyncio.run(scenario())


def test_reload_cancellation_during_other_section_cleans_uncommitted_tts_and_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module

    initial, runtime = _runtime(
        tmp_path,
        {"tts": {"enabled": False, "backend": "text_only", "language": "zh"}},
    )
    replacement_tts_closed = asyncio.Event()
    replacement_router_closed = asyncio.Event()
    behavior_entered = asyncio.Event()
    behavior_release = asyncio.Event()

    class ReadyBackend:
        async def start(self) -> EngineHealth:
            return EngineHealth("replacement", True, "ready")

        async def health(self) -> EngineHealth:
            return EngineHealth("replacement", True, "ready")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            replacement_tts_closed.set()

    class ReplacementRouter:
        def diagnostics(self, _task: str):
            return {"ready": True}

        async def aclose(self) -> None:
            replacement_router_closed.set()

    replacement_tts = TTSCoordinator(ReadyBackend())
    replacement_router = ReplacementRouter()
    monkeypatch.setattr(runtime_module, "validate_runtime_configuration", lambda _value: None)
    monkeypatch.setattr(runtime_module, "_build_tts", lambda _value: replacement_tts)
    monkeypatch.setattr(
        runtime_module.ModelRouter,
        "from_mapping",
        lambda _values: replacement_router,
    )

    async def blocked_behavior(_values: object, *, on_commit=None) -> None:
        del on_commit
        behavior_entered.set()
        await behavior_release.wait()

    monkeypatch.setattr(runtime, "_apply_behavior_configuration", blocked_behavior)
    updated = _configuration(
        tmp_path,
        {
            "tts": {"enabled": False, "backend": "text_only", "language": "ja"},
            "llm": {"channels": []},
            "behavior": {"enabled": False, "probability": 0.2},
        },
    )
    old_backend = runtime.tts.backend
    old_router = runtime.router

    async def scenario() -> None:
        task = asyncio.create_task(runtime.apply_configuration(updated))
        await behavior_entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        behavior_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(replacement_tts_closed.wait(), timeout=1.0)
        await asyncio.wait_for(replacement_router_closed.wait(), timeout=1.0)
        assert runtime.configuration.values["behavior"] == updated.values["behavior"]
        assert runtime.configuration.values["tts"] == initial.values["tts"]
        assert runtime.configuration.values.get("llm") == initial.values.get("llm")
        assert runtime.tts.backend is old_backend
        assert runtime.router is old_router
        assert runtime.conversation.router is old_router
        assert not runtime._reload_cleanup_tasks
        await runtime.close()

    asyncio.run(scenario())


def test_reload_exception_cleanup_cancellation_still_resumes_old_startup_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module

    initial, runtime = _runtime(
        tmp_path,
        {
            "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
            "asr": {"enabled": False},
            "scheduler": {"enabled": False, "activity": {"enabled": False}},
            "watcher": {"enabled": False},
            "behavior": {"enabled": False},
            "config": {"reload": {"enabled": False}},
        },
    )
    old_started = asyncio.Event()
    old_restarted = asyncio.Event()
    replacement_close_started = asyncio.Event()
    replacement_close_release = asyncio.Event()
    replacement_closed = asyncio.Event()

    class OldBackend:
        def __init__(self) -> None:
            self.start_calls = 0

        async def start(self) -> EngineHealth:
            self.start_calls += 1
            if self.start_calls == 1:
                old_started.set()
                await asyncio.Event().wait()
            old_restarted.set()
            return EngineHealth("old", True, "ready")

        async def health(self) -> EngineHealth:
            return EngineHealth("old", True, "ready")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            return None

    class ReplacementBackend:
        async def start(self) -> EngineHealth:
            return EngineHealth("replacement", True, "ready")

        async def health(self) -> EngineHealth:
            return EngineHealth("replacement", True, "ready")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            replacement_close_started.set()
            await replacement_close_release.wait()
            replacement_closed.set()

    async def fail_behavior(_values: object, *, on_commit=None) -> None:
        del on_commit
        raise RuntimeError("behavior apply failed")

    old_backend = OldBackend()
    replacement = TTSCoordinator(ReplacementBackend())
    runtime.tts.backend = old_backend
    monkeypatch.setattr(runtime_module, "_build_tts", lambda _configuration: replacement)
    monkeypatch.setattr(runtime, "_apply_behavior_configuration", fail_behavior)
    updated = _configuration(
        tmp_path,
        {
            **initial.values,
            "tts": {"enabled": False, "backend": "text_only", "language": "ja"},
            "behavior": {"enabled": False, "probability": 0.2},
        },
    )

    async def scenario() -> None:
        await runtime.start_background()
        await old_started.wait()
        reload_task = asyncio.create_task(runtime.apply_configuration(updated))
        await replacement_close_started.wait()
        reload_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reload_task
        await asyncio.wait_for(old_restarted.wait(), timeout=1.0)
        assert runtime.configuration is initial
        assert runtime.tts.backend is old_backend
        assert runtime._reload_cleanup_tasks
        replacement_close_release.set()
        await asyncio.wait_for(replacement_closed.wait(), timeout=1.0)
        await asyncio.wait_for(runtime._drain_reload_cleanups(), timeout=1.0)
        assert not runtime._reload_cleanup_tasks
        await runtime.close()

    asyncio.run(scenario())


def test_reload_cancellation_preserves_all_completed_section_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial, runtime = _runtime(
        tmp_path,
        {"memory": {"enabled": True, "recall_limit": 4}},
    )
    behavior_entered = asyncio.Event()
    behavior_release = asyncio.Event()

    async def blocked_behavior(_values: object, *, on_commit=None) -> None:
        del on_commit
        behavior_entered.set()
        await behavior_release.wait()

    monkeypatch.setattr(runtime, "_apply_behavior_configuration", blocked_behavior)
    updated = _configuration(
        tmp_path,
        {
            "memory": {"enabled": True, "recall_limit": 7},
            "behavior": {"enabled": False, "probability": 0.2},
        },
    )

    async def scenario() -> None:
        reload_task = asyncio.create_task(runtime.apply_configuration(updated))
        await behavior_entered.wait()
        reload_task.cancel()
        await asyncio.sleep(0)
        assert reload_task.done() is False
        behavior_release.set()
        with pytest.raises(asyncio.CancelledError):
            await reload_task
        assert runtime.memory.settings.recall_limit == 7
        assert runtime.configuration.values["memory"] == updated.values["memory"]
        assert runtime.configuration.values["behavior"] == updated.values["behavior"]
        await runtime.close()

    asyncio.run(scenario())


def test_behavior_section_records_commit_before_blocking_lifecycle_await(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial, runtime = _runtime(
        tmp_path,
        {
            "behavior": {"enabled": False, "probability": 0.1},
            "scheduler": {"enabled": False, "activity": {"enabled": False}},
            "watcher": {"enabled": False},
            "config": {"reload": {"enabled": False}},
        },
    )
    start_entered = asyncio.Event()
    start_release = asyncio.Event()
    lifecycle_completed = False

    async def blocked_start() -> None:
        nonlocal lifecycle_completed
        start_entered.set()
        await start_release.wait()
        runtime.behavior._task = asyncio.create_task(asyncio.Event().wait())
        lifecycle_completed = True

    monkeypatch.setattr(runtime.behavior, "start", blocked_start)
    updated = _configuration(
        tmp_path,
        {
            **initial.values,
            "behavior": {
                "enabled": True,
                "probability": 0.75,
                "movement": {"enabled": False},
            },
        },
    )

    async def scenario() -> None:
        await runtime.start_background()
        reload_task = asyncio.create_task(runtime.apply_configuration(updated))
        await start_entered.wait()
        reload_task.cancel()
        await asyncio.sleep(0)
        assert reload_task.done() is False
        start_release.set()
        with pytest.raises(asyncio.CancelledError):
            await reload_task
        assert runtime.configuration.values["behavior"] == updated.values["behavior"]
        assert runtime.behavior._probability == 0.75
        assert lifecycle_completed is True
        assert runtime.behavior.status()["running"] is True
        repeated = await runtime.apply_configuration(updated)
        assert repeated["status"] == "unchanged"
        await runtime.close()

    asyncio.run(scenario())


def test_unsupported_sections_remain_pending_across_repeated_reload(
    tmp_path: Path,
) -> None:
    initial, runtime = _runtime(
        tmp_path,
        {"memory": {"enabled": True, "recall_limit": 4}},
    )
    updated = _configuration(
        tmp_path,
        {
            "memory": {"enabled": True, "recall_limit": 7},
            "web": {
                "local_api": {
                    "enabled": False,
                    "host": "127.0.0.1",
                    "port": 0,
                    "events": True,
                    "event_limit": 32,
                }
            },
        },
    )

    async def scenario() -> None:
        first = await runtime.apply_configuration(updated)
        assert first["status"] == "restart_required"
        assert first["applied_sections"] == ("memory",)
        assert first["restart_sections"] == ("web",)
        assert runtime.configuration.values["memory"] == updated.values["memory"]
        assert runtime.configuration.values.get("web") == initial.values.get("web")

        second = await runtime.apply_configuration(updated)
        assert second["status"] == "restart_required"
        assert second["changed_sections"] == ("web",)
        assert second["restart_sections"] == ("web",)
        assert second["status"] != "unchanged"
        await runtime.close()

    asyncio.run(scenario())


def test_rendering_reload_finishes_dispatch_before_cancellation_is_propagated(
    tmp_path: Path,
) -> None:
    initial_root = tmp_path / "initial-resources"
    updated_root = tmp_path / "updated-resources"
    initial, runtime = _runtime(
        tmp_path,
        {"rendering": {"backend": "auto", "resource_root": str(initial_root)}},
    )
    reload_entered = asyncio.Event()
    reload_release = asyncio.Event()
    reload_applied = False

    class RendererHost:
        def reload_resources(self, _resource_root, **_kwargs):
            async def apply_reload():
                nonlocal reload_applied
                reload_entered.set()
                await reload_release.wait()
                reload_applied = True
                return {"status": "reloaded"}

            return apply_reload()

    runtime.pet_controller.attach(RendererHost())
    updated = _configuration(
        tmp_path,
        {"rendering": {"backend": "auto", "resource_root": str(updated_root)}},
    )

    async def scenario() -> None:
        reload_task = asyncio.create_task(runtime.apply_configuration(updated))
        await reload_entered.wait()
        reload_task.cancel()
        await asyncio.sleep(0)
        assert reload_task.done() is False
        reload_release.set()
        with pytest.raises(asyncio.CancelledError):
            await reload_task
        assert reload_applied is True
        assert runtime.configuration.values["rendering"] == updated.values["rendering"]
        assert runtime.inventory.root == updated_root.resolve()
        repeated = await runtime.apply_configuration(updated)
        assert repeated["status"] == "unchanged"
        assert runtime.configuration.path == updated.path
        assert initial is not runtime.configuration
        await runtime.close()

    asyncio.run(scenario())


def test_reload_cancellation_during_asr_start_closes_uncommitted_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module

    initial, runtime = _runtime(
        tmp_path,
        {"asr": {"enabled": False, "backend": "sensevoice", "language": "auto"}},
    )
    replacement_started = asyncio.Event()
    replacement_closed = asyncio.Event()

    class ReplacementASR:
        enabled = False

        async def start(self) -> ASRHealth:
            replacement_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            replacement_closed.set()

    replacement = ReplacementASR()
    monkeypatch.setattr(runtime_module, "validate_runtime_configuration", lambda _value: None)
    monkeypatch.setattr(runtime_module, "_build_asr", lambda _value: replacement)
    updated = _configuration(
        tmp_path,
        {"asr": {"enabled": False, "backend": "sensevoice", "language": "zh"}},
    )
    old_asr = runtime.asr

    async def scenario() -> None:
        task = asyncio.create_task(runtime.apply_configuration(updated))
        await replacement_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(replacement_closed.wait(), timeout=1.0)
        assert runtime.configuration is initial
        assert runtime.asr is old_asr
        assert not runtime._reload_cleanup_tasks
        await runtime.close()

    asyncio.run(scenario())


def test_asr_tool_uses_hot_reloaded_runtime_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module

    initial, runtime = _runtime(
        tmp_path,
        {"asr": {"enabled": False, "backend": "sensevoice", "language": "auto"}},
    )
    transcribe_calls = 0

    class ReplacementASR:
        enabled = False
        max_audio_bytes = 4096
        language = "zh"

        async def start(self) -> ASRHealth:
            return ASRHealth(
                "disabled",
                False,
                "sensevoice",
                "SenseVoiceSmall",
                False,
                "cpu",
                "zh",
            )

        def has_active_tasks(self) -> bool:
            return False

        def diagnostics(self):
            return {"status": "disabled", "available": False, "ready": False}

        async def transcribe(self, audio: bytes, **_kwargs: object):
            nonlocal transcribe_calls
            assert audio.startswith(b"RIFF")
            transcribe_calls += 1
            return {
                "request_id": "hot-reload-asr",
                "text": "新服务转写",
                "language": "zh",
                "confidence": None,
                "confidence_available": False,
                "duration_ms": 3,
            }

        async def aclose(self) -> None:
            return None

    replacement = ReplacementASR()
    monkeypatch.setattr(runtime_module, "validate_runtime_configuration", lambda _value: None)
    monkeypatch.setattr(runtime_module, "_build_asr", lambda _value: replacement)
    updated = _configuration(
        tmp_path,
        {"asr": {"enabled": False, "backend": "sensevoice", "language": "zh"}},
    )

    async def scenario() -> None:
        result = await runtime.apply_configuration(updated)
        assert result["status"] == "reloaded"
        wav = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 32
        tool = runtime.registry.require("system:transcribe_audio")
        transcription = await tool.handler(
            {
                "audio_base64": base64.b64encode(wav).decode("ascii"),
                "audio_format": "wav",
                "language": "zh",
            },
            ToolCallContext("profile", "session", "turn"),
        )
        assert transcription["status"] == "completed"
        assert transcription["text"] == "新服务转写"
        assert transcribe_calls == 1
        assert runtime.asr is replacement
        assert runtime.configuration is updated
        assert initial is not runtime.configuration
        await runtime.close()

    asyncio.run(scenario())


def test_reload_cancellation_after_tts_swap_keeps_new_backend_and_tracks_old_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module

    initial, runtime = _runtime(
        tmp_path,
        {"tts": {"enabled": False, "backend": "text_only", "language": "zh"}},
    )
    old_close_started = asyncio.Event()
    old_close_release = asyncio.Event()
    old_close_finished = asyncio.Event()

    class OldBackend:
        async def aclose(self) -> None:
            old_close_started.set()
            await old_close_release.wait()
            old_close_finished.set()

    class NewBackend:
        def __init__(self) -> None:
            self.close_calls = 0

        async def start(self) -> EngineHealth:
            return EngineHealth("replacement", True, "ready")

        async def health(self) -> EngineHealth:
            return EngineHealth("replacement", True, "ready")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            self.close_calls += 1

    old_backend = OldBackend()
    new_backend = NewBackend()
    replacement = TTSCoordinator(new_backend)
    runtime.tts.backend = old_backend
    monkeypatch.setattr(runtime_module, "_build_tts", lambda _configuration: replacement)
    updated = _configuration(
        tmp_path,
        {"tts": {"enabled": False, "backend": "text_only", "language": "ja"}},
    )

    async def scenario() -> None:
        task = asyncio.create_task(runtime.apply_configuration(updated))
        await old_close_started.wait()
        assert runtime.tts.backend is new_backend
        assert runtime.configuration is updated
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert new_backend.close_calls == 0
        assert runtime._reload_cleanup_tasks
        first_close_waiter = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert first_close_waiter.done() is False
        first_close_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close_waiter
        second_close_waiter = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert second_close_waiter.done() is False
        old_close_release.set()
        await asyncio.wait_for(old_close_finished.wait(), timeout=1.0)
        await asyncio.wait_for(second_close_waiter, timeout=1.0)
        assert not runtime._reload_cleanup_tasks
        assert runtime._reload_cleanup_required is False
        assert new_backend.close_calls == 1

    asyncio.run(scenario())


def test_repeated_cancellation_cannot_interrupt_startup_generation_cleanup(
    tmp_path: Path,
) -> None:
    _initial, runtime = _runtime(tmp_path, {})

    async def exercise(attribute: str, cancel_method) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def startup() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                entered.set()
                await release.wait()

        startup_task = asyncio.create_task(startup())
        setattr(runtime, attribute, startup_task)
        cancellation = asyncio.create_task(cancel_method())
        await entered.wait()
        cancellation.cancel()
        await asyncio.sleep(0)
        cancellation.cancel()
        await asyncio.sleep(0)
        assert startup_task.done() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancellation
        assert startup_task.done() is True
        assert getattr(runtime, attribute) is None

    async def scenario() -> None:
        await exercise("_tts_start_task", runtime._cancel_tts_initialization)
        await exercise("_asr_start_task", runtime._cancel_asr_initialization)
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_close_during_reload_does_not_restart_paused_model_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module

    initial, runtime = _runtime(
        tmp_path,
        {
            "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
            "asr": {"enabled": False},
            "scheduler": {"enabled": False, "activity": {"enabled": False}},
            "watcher": {"enabled": False},
            "behavior": {"enabled": False},
            "config": {"reload": {"enabled": False}},
        },
    )
    old_started = asyncio.Event()
    replacement_started = asyncio.Event()
    replacement_close_started = asyncio.Event()
    replacement_close_release = asyncio.Event()
    replacement_closed = asyncio.Event()

    class OldBackend:
        def __init__(self) -> None:
            self.start_calls = 0

        async def start(self) -> EngineHealth:
            self.start_calls += 1
            old_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def health(self) -> EngineHealth:
            return EngineHealth("old", False, "loading")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            return None

    class ReplacementBackend:
        async def start(self) -> EngineHealth:
            replacement_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def health(self) -> EngineHealth:
            return EngineHealth("replacement", False, "loading")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            replacement_close_started.set()
            await replacement_close_release.wait()
            replacement_closed.set()

    old_backend = OldBackend()
    replacement = TTSCoordinator(ReplacementBackend())
    runtime.tts.backend = old_backend
    monkeypatch.setattr(runtime_module, "_build_tts", lambda _configuration: replacement)
    updated = _configuration(
        tmp_path,
        {**initial.values, "tts": {"enabled": False, "backend": "text_only", "language": "ja"}},
    )

    async def scenario() -> None:
        await runtime.start_background()
        await old_started.wait()
        reload_task = asyncio.create_task(runtime.apply_configuration(updated))
        await replacement_started.wait()
        close_task = asyncio.create_task(runtime.close())
        await asyncio.wait_for(replacement_close_started.wait(), timeout=1.0)
        assert close_task.done() is False
        replacement_close_release.set()
        await asyncio.wait_for(close_task, timeout=1.0)
        with pytest.raises(asyncio.CancelledError):
            await reload_task
        await asyncio.wait_for(replacement_closed.wait(), timeout=1.0)
        assert old_backend.start_calls == 1
        assert runtime._tts_start_task is None
        assert runtime._asr_start_task is None
        assert runtime.configuration is initial
        assert runtime._closed is True

    asyncio.run(scenario())
