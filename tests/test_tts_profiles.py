from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.runtime import ApplicationRuntime, _build_tts, validate_runtime_configuration
from config.loader import LoadedConfiguration
from core.events.types import ConversationContext
from core.tts.contracts import EngineHealth, SpeechChunk, SpeechRequest
from services.tts.backend import TextOnlyBackend
from services.tts.coordinator import TTSCoordinator
from services.tts.router import TTSProfile, TTSProfileRouter


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _Backend:
    def __init__(self, name: str, *, fail: bool = False, partial: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.partial = partial
        self.calls = 0

    async def health(self) -> EngineHealth:
        return EngineHealth(self.name, not self.fail, self.name)

    async def stream(self, request: SpeechRequest):
        self.calls += 1
        if self.partial:
            yield SpeechChunk(request.request_id, self.name.encode(), 24000, 1)
            raise RuntimeError(self.name)
        if self.fail:
            raise RuntimeError(self.name)
        yield SpeechChunk(request.request_id, self.name.encode(), 24000, 1, is_final=True)


def _request(language: str = "zh") -> SpeechRequest:
    return SpeechRequest("profile-request", "你好。", language=language)


def test_profile_router_selects_language_mapping_and_preserves_legacy_backend_shape() -> None:
    zh = _Backend("zh")
    ja = _Backend("ja")
    router = TTSProfileRouter(
        {
            "zh-main": TTSProfile("zh-main", zh, languages=frozenset({"zh"}), priority=1),
            "ja-main": TTSProfile("ja-main", ja, languages=frozenset({"jp"}), priority=1),
        },
        default_profile="zh-main",
        language_profiles={"ja": "ja-main"},
    )

    async def scenario() -> tuple[bytes, bytes]:
        first = [chunk async for chunk in router.stream(_request("zh"))]
        second = [chunk async for chunk in router.stream(_request("ja"))]
        return first[0].data, second[0].data

    assert asyncio.run(scenario()) == (b"zh", b"ja")
    assert router.backend is zh
    assert router.active_profile == "zh-main"


def test_profile_router_selects_role_mapping_before_global_profile() -> None:
    global_voice = _Backend("global")
    character_voice = _Backend("character")
    router = TTSProfileRouter(
        {
            "global": TTSProfile("global", global_voice),
            "character": TTSProfile("character", character_voice, roles=frozenset({"maid"})),
        },
        default_profile="global",
        role_profiles={"maid": "character"},
    )
    router.select("global")

    async def scenario() -> list[SpeechChunk]:
        return [
            chunk
            async for chunk in router.stream(SpeechRequest("role-request", "你好。", role="maid"))
        ]

    assert asyncio.run(scenario())[0].data == b"character"
    assert global_voice.calls == 0
    assert character_voice.calls == 1


def test_coordinator_keeps_role_profile_snapshot_during_hot_switch() -> None:
    global_voice = _Backend("global")
    character_voice = _Backend("character")
    profile_router = TTSProfileRouter(
        {
            "global": TTSProfile("global", global_voice),
            "character": TTSProfile("character", character_voice),
        },
        default_profile="global",
        role_profiles={"maid": "character"},
    )
    coordinator = TTSCoordinator(profile_router)
    context = ConversationContext("profile", "local", "role-pinned", 1)

    async def scenario() -> None:
        assert coordinator.pin_context(context, language="zh", role="maid") == (
            "zh",
            "character",
        )
        profile_router.reconfigure(
            {
                "global": TTSProfile("global", global_voice),
                "character": TTSProfile("character", character_voice),
            },
            default_profile="global",
            role_profiles={"maid": "global"},
        )
        await coordinator.enqueue_text(
            context,
            "热切换后仍使用原角色。",
            language="zh",
            role="maid",
            flush=True,
        )
        coordinator.release_context(context)

    asyncio.run(scenario())

    assert character_voice.calls == 1
    assert global_voice.calls == 0


def test_coordinator_allows_explicit_role_override_for_pinned_turn() -> None:
    default_voice = _Backend("default")
    override_voice = _Backend("override")
    profile_router = TTSProfileRouter(
        {
            "default": TTSProfile("default", default_voice),
            "override": TTSProfile("override", override_voice),
        },
        default_profile="default",
        role_profiles={"assistant": "default", "narrator": "override"},
    )
    coordinator = TTSCoordinator(profile_router)
    context = ConversationContext("profile", "local", "role-override", 1)

    async def scenario() -> None:
        coordinator.pin_context(context, language="zh", role="assistant")
        await coordinator.enqueue_text(
            context,
            "工具明确指定旁白角色。",
            language="zh",
            role="narrator",
            flush=True,
        )
        coordinator.release_context(context)

    asyncio.run(scenario())

    assert override_voice.calls == 1
    assert default_voice.calls == 0


def test_coordinator_pins_language_and_profile_for_a_turn() -> None:
    class RecordingBackend(_Backend):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.requests: list[SpeechRequest] = []

        async def stream(self, request: SpeechRequest):
            self.requests.append(request)
            async for chunk in super().stream(request):
                yield chunk

    first = RecordingBackend("first")
    second = RecordingBackend("second")
    profile_router = TTSProfileRouter(
        {
            "first": TTSProfile("first", first, languages=frozenset({"zh"})),
            "second": TTSProfile("second", second, languages=frozenset({"jp"})),
        },
        default_profile="first",
        language_profiles={"ja": "second"},
    )
    coordinator = TTSCoordinator(profile_router)
    context = ConversationContext("profile", "local", "pinned", 1)

    async def scenario() -> None:
        await coordinator.enqueue_text(context, "第一句。", language="zh")
        profile_router.select("second", language="ja")
        await coordinator.enqueue_text(context, "第二句。", language="ja", flush=True)

    asyncio.run(scenario())

    assert [request.text for request in first.requests] == ["第一句。", "第二句。"]
    assert [request.language for request in first.requests] == ["zh", "zh"]
    assert [request.profile_id for request in first.requests] == ["first", "first"]
    assert not second.requests


def test_clear_context_routes_preserves_an_unfinished_language_snapshot() -> None:
    class RecordingBackend(_Backend):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.requests: list[SpeechRequest] = []

        async def stream(self, request: SpeechRequest):
            self.requests.append(request)
            async for chunk in super().stream(request):
                yield chunk

    first = RecordingBackend("first")
    second = RecordingBackend("second")
    profile_router = TTSProfileRouter(
        {
            "first": TTSProfile("first", first, languages=frozenset({"zh"})),
            "second": TTSProfile("second", second, languages=frozenset({"jp"})),
        },
        default_profile="first",
        language_profiles={"ja": "second"},
    )
    coordinator = TTSCoordinator(profile_router)
    context = ConversationContext("profile", "local", "unfinished", 1)

    async def scenario() -> None:
        await coordinator.enqueue_text(context, "未结束", language="zh")
        # 没有活动合成 task 的间隙仍可能发生配置热切换；未完成尾部
        # 必须保持首次入队时的语言/profile。
        coordinator.clear_context_routes()
        await coordinator.enqueue_text(context, "。", language="ja", flush=True)

    asyncio.run(scenario())

    assert [(request.text, request.language, request.profile_id) for request in first.requests] == [
        ("未结束。", "zh", "first")
    ]
    assert second.requests == []


def test_profile_router_falls_back_and_cools_failed_profile() -> None:
    clock = _Clock()
    broken = _Backend("broken", fail=True)
    fallback = _Backend("fallback")
    router = TTSProfileRouter(
        {
            "primary": TTSProfile(
                "primary", broken, languages=frozenset({"zh"}), cooldown_seconds=5
            ),
            "fallback": TTSProfile("fallback", fallback, languages=frozenset({"*"}), priority=-1),
        },
        default_profile="primary",
        fallback_profiles=("fallback",),
        clock=clock,
    )

    async def run_once() -> list[bytes]:
        return [chunk.data async for chunk in router.stream(_request())]

    assert asyncio.run(run_once()) == [b"fallback"]
    assert broken.calls == 1
    assert asyncio.run(run_once()) == [b"fallback"]
    assert broken.calls == 1
    clock.value = 5.0
    assert asyncio.run(run_once()) == [b"fallback"]
    assert broken.calls == 2


def test_profile_router_rewrites_profile_id_for_each_fallback_backend() -> None:
    """后备后端必须收到自己的 profile 标识，而不是失败主 profile。"""

    class RecordingBackend(_Backend):
        def __init__(self, name: str, *, fail: bool = False) -> None:
            super().__init__(name, fail=fail)
            self.request_profile_ids: list[str] = []

        async def stream(self, request: SpeechRequest):
            self.request_profile_ids.append(request.profile_id)
            async for chunk in super().stream(request):
                yield chunk

    primary = RecordingBackend("primary", fail=True)
    fallback = RecordingBackend("fallback")
    router = TTSProfileRouter(
        {
            "primary": TTSProfile("primary", primary, cooldown_seconds=0),
            "fallback": TTSProfile("fallback", fallback),
        },
        default_profile="primary",
        fallback_profiles=("fallback",),
    )

    async def scenario() -> list[bytes]:
        request = SpeechRequest("fallback-profile-id", "你好。", profile_id="primary")
        return [chunk.data async for chunk in router.stream(request)]

    assert asyncio.run(scenario()) == [b"fallback"]
    assert primary.request_profile_ids == ["primary"]
    assert fallback.request_profile_ids == ["fallback"]


def test_profile_router_retries_full_fallback_chain_when_all_profiles_are_cooling() -> None:
    """共同故障后仍应尝试已配置的后备 profile，而不是只重试主 profile。"""

    clock = _Clock()
    primary = _Backend("primary", fail=True)
    fallback = _Backend("fallback", fail=True)
    router = TTSProfileRouter(
        {
            "primary": TTSProfile("primary", primary, cooldown_seconds=30),
            "fallback": TTSProfile("fallback", fallback, cooldown_seconds=30),
        },
        default_profile="primary",
        fallback_profiles=("fallback",),
        clock=clock,
    )

    async def run_once() -> None:
        with pytest.raises(RuntimeError, match="all TTS profiles failed"):
            async for _chunk in router.stream(_request()):
                pass

    asyncio.run(run_once())
    assert primary.calls == 1
    assert fallback.calls == 1

    # 两个 profile 都还在冷却时，第二次请求仍需完整走一次回退链。
    asyncio.run(run_once())
    assert primary.calls == 2
    assert fallback.calls == 2


def test_profile_router_does_not_duplicate_partial_audio_on_fallback() -> None:
    broken = _Backend("broken", partial=True)
    fallback = _Backend("fallback")
    router = TTSProfileRouter(
        {
            "primary": TTSProfile("primary", broken, cooldown_seconds=0),
            "fallback": TTSProfile("fallback", fallback),
        },
        default_profile="primary",
        fallback_profiles=("fallback",),
    )

    async def scenario() -> None:
        stream = router.stream(_request())
        assert (await anext(stream)).data == b"broken"
        with pytest.raises(RuntimeError, match="broken"):
            await anext(stream)

    asyncio.run(scenario())
    assert fallback.calls == 0


def test_profile_router_rejects_mismatched_request_id_before_fallback() -> None:
    class Broken(_Backend):
        async def stream(self, request: SpeechRequest):
            self.calls += 1
            yield SpeechChunk("other-request", b"audio", 24000, 1, is_final=True)

    broken = Broken("broken")
    fallback = _Backend("fallback")
    router = TTSProfileRouter(
        {
            "primary": TTSProfile("primary", broken, cooldown_seconds=0),
            "fallback": TTSProfile("fallback", fallback),
        },
        default_profile="primary",
        fallback_profiles=("fallback",),
    )

    async def scenario() -> list[SpeechChunk]:
        return [chunk async for chunk in router.stream(_request())]

    chunks = asyncio.run(scenario())
    assert [chunk.data for chunk in chunks] == [b"fallback"]
    assert broken.calls == 1
    assert fallback.calls == 1


def test_profile_router_rejects_explicit_disabled_profile() -> None:
    disabled = _Backend("disabled")
    router = TTSProfileRouter(
        {
            "disabled": TTSProfile("disabled", disabled, enabled=False),
            "fallback": TTSProfile("fallback", _Backend("fallback")),
        },
        default_profile="fallback",
    )

    async def scenario() -> None:
        request = SpeechRequest("explicit", "你好。", profile_id="disabled")
        with pytest.raises(RuntimeError, match="profile is unavailable"):
            [chunk async for chunk in router.stream(request)]

    asyncio.run(scenario())
    assert disabled.calls == 0


def test_profile_router_falls_back_when_primary_ends_without_audio() -> None:
    class EmptyAudioBackend(_Backend):
        async def stream(self, request: SpeechRequest):
            self.calls += 1
            yield SpeechChunk(request.request_id, b"", 24000, 1, is_final=True)

    empty = EmptyAudioBackend("empty")
    fallback = _Backend("fallback")
    router = TTSProfileRouter(
        {
            "primary": TTSProfile("primary", empty, cooldown_seconds=5),
            "fallback": TTSProfile("fallback", fallback),
        },
        default_profile="primary",
        fallback_profiles=("fallback",),
    )

    async def scenario() -> list[bytes]:
        return [chunk.data async for chunk in router.stream(_request())]

    assert asyncio.run(scenario()) == [b"fallback"]
    assert empty.calls == 1
    assert fallback.calls == 1
    assert router.diagnostics()["profiles"][0]["failures"] == 1


def test_text_only_profile_is_allowed_to_complete_without_audio() -> None:
    backend = TextOnlyBackend()
    profile = TTSProfile("text", backend)

    async def scenario() -> list[SpeechChunk]:
        return [chunk async for chunk in TTSProfileRouter({"text": profile}).stream(_request())]

    assert asyncio.run(scenario()) == []
    assert profile.requires_audio is False


def test_profile_router_select_and_reconfigure_are_atomic() -> None:
    first = _Backend("first")
    second = _Backend("second")
    router = TTSProfileRouter({"first": first}, default_profile="first")
    assert router.select("first") == "first"
    with pytest.raises(KeyError):
        router.select("missing")
    assert router.select(None) == "first"
    retired = router.reconfigure(
        {"second": TTSProfile("second", second, languages=frozenset({"zh"}))},
        default_profile="second",
    )
    assert retired == (first,)
    assert router.profile_ids() == ("second",)


def test_profile_router_rejects_disabled_selection_and_language_mapping() -> None:
    backend = _Backend("disabled")
    router = TTSProfileRouter(
        {"disabled": TTSProfile("disabled", backend, enabled=False)},
    )

    with pytest.raises(ValueError, match="disabled"):
        router.select("disabled")
    with pytest.raises(ValueError, match="disabled"):
        router.select("disabled", language="zh")


def test_profile_router_rejects_disabled_fallback_profile() -> None:
    profiles = {
        "disabled": TTSProfile("disabled", _Backend("disabled"), enabled=False),
        "active": TTSProfile("active", _Backend("active")),
    }
    router = TTSProfileRouter(profiles)

    with pytest.raises(ValueError, match="fallback"):
        router.reconfigure(
            profiles,
            default_profile="active",
            fallback_profiles=("disabled",),
        )


def test_coordinator_rejects_disabled_explicit_voice_profile() -> None:
    backend = _Backend("disabled")
    coordinator = TTSCoordinator(
        TTSProfileRouter({"disabled": TTSProfile("disabled", backend, enabled=False)})
    )

    with pytest.raises(ValueError, match="voice profile"):
        asyncio.run(
            coordinator.enqueue_text(
                ConversationContext("p", "s", "disabled-voice", 1),
                "禁用音色。",
                profile_id="disabled",
                flush=True,
            )
        )


def test_profile_router_health_uses_first_available_profile() -> None:
    broken = _Backend("broken", fail=True)
    healthy = _Backend("healthy")
    router = TTSProfileRouter(
        {"broken": broken, "healthy": healthy},
        default_profile="broken",
        fallback_profiles=("healthy",),
    )
    health = asyncio.run(router.health())
    assert health.available is True
    assert health.engine == "tts-router:healthy"


def test_profile_metadata_rejects_ambiguous_priority_and_boolean_values() -> None:
    backend = _Backend("backend")
    with pytest.raises(ValueError, match="priority"):
        TTSProfile("bad", backend, priority=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        TTSProfile("bad", backend, enabled="maybe")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="profile id"):
        TTSProfile("bad\nvoice", backend)


def test_coordinator_accepts_explicit_voice_profile_without_global_switch() -> None:
    requests: list[SpeechRequest] = []

    class RecordingBackend(_Backend):
        async def stream(self, request: SpeechRequest):
            requests.append(request)
            async for chunk in super().stream(request):
                yield chunk

    first = _Backend("first")
    second = RecordingBackend("second")
    router = TTSProfileRouter(
        {
            "first": TTSProfile("first", first),
            "second": TTSProfile("second", second),
        },
        default_profile="first",
    )
    coordinator = TTSCoordinator(router)

    asyncio.run(
        coordinator.enqueue_text(
            ConversationContext("p", "s", "voice", 1),
            "指定音色。",
            language="zh",
            profile_id="second",
            flush=True,
        )
    )

    assert requests and requests[0].profile_id == "second"
    assert second.calls == 1
    assert first.calls == 0
    assert router.selected_profile == ""


def test_coordinator_rejects_unknown_explicit_voice_profile() -> None:
    coordinator = TTSCoordinator(
        TTSProfileRouter({"first": TTSProfile("first", _Backend("first"))})
    )
    with pytest.raises(ValueError, match="voice profile"):
        asyncio.run(
            coordinator.enqueue_text(
                ConversationContext("p", "s", "missing-voice", 1),
                "不存在的音色。",
                profile_id="missing",
                flush=True,
            )
        )


def test_runtime_builds_profile_router_without_changing_legacy_backend(tmp_path) -> None:
    coordinator = _build_tts(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {
                    "enabled": True,
                    "backend": "text_only",
                    "profiles": {
                        "primary": {"backend": "text_only", "languages": ["zh"]},
                        "fallback": {
                            "backend": "text_only",
                            "languages": ["*"],
                            "priority": -1,
                        },
                    },
                    "routing": {
                        "default_profile": "primary",
                        "language_profiles": {"zh": "primary"},
                        "fallback_profiles": ["fallback"],
                    },
                }
            },
        )
    )
    try:
        assert isinstance(coordinator, TTSCoordinator)
        assert isinstance(coordinator.backend, TTSProfileRouter)
        assert coordinator.backend.profile_ids() == ("primary", "fallback")
        assert coordinator.backend.active_profile == "primary"
    finally:
        asyncio.run(coordinator.aclose())


def test_runtime_builds_role_profile_routing(tmp_path) -> None:
    coordinator = _build_tts(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {
                    "enabled": True,
                    "backend": "text_only",
                    "language": "zh",
                    "profiles": {
                        "default": {"backend": "text_only"},
                        "narrator": {"backend": "text_only", "roles": ["narrator"]},
                    },
                    "routing": {
                        "default_profile": "default",
                        "role_profiles": {"narrator": "narrator"},
                    },
                }
            },
        )
    )
    try:
        assert isinstance(coordinator.backend, TTSProfileRouter)
        assert coordinator.backend.profile_for("zh", role="narrator") == "narrator"
        assert coordinator.backend.diagnostics()["role_profiles"] == {"narrator": "narrator"}
    finally:
        asyncio.run(coordinator.aclose())


def test_runtime_selects_tts_profile_and_language_without_replacing_stream_router() -> None:
    first = _Backend("first")
    second = _Backend("second")
    profile_router = TTSProfileRouter(
        {
            "first": TTSProfile("first", first, languages=frozenset({"zh"})),
            "second": TTSProfile("second", second, languages=frozenset({"jp"})),
        },
        default_profile="first",
    )
    runtime = object.__new__(ApplicationRuntime)
    runtime.tts = SimpleNamespace(backend=profile_router)
    runtime.conversation = SimpleNamespace(tts_language="zh")

    selected = runtime.select_tts_profile("second", language="ja")
    assert selected["status"] == "updated"
    assert selected["profile"] == "second"
    assert selected["language"] == "jp"
    assert runtime.conversation.tts_language == "jp"
    assert profile_router.active_profile == "second"
    assert runtime.set_tts_language("en")["language_protocol"] == "en"


def test_runtime_rejects_unknown_tts_profile_without_mutating_language() -> None:
    profile_router = TTSProfileRouter({"first": TTSProfile("first", _Backend("first"))})
    runtime = object.__new__(ApplicationRuntime)
    runtime.tts = SimpleNamespace(backend=profile_router)
    runtime.conversation = SimpleNamespace(tts_language="zh")

    result = runtime.select_tts_profile("missing", language="ja")
    assert result["status"] == "unavailable"
    assert runtime.conversation.tts_language == "zh"


def test_runtime_rejects_unknown_profile_route_before_database_creation(tmp_path) -> None:
    from config.loader import ConfigurationError

    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "tts": {
                "enabled": True,
                "profiles": {"safe": {"backend": "text_only"}},
                "routing": {"default_profile": "missing"},
            }
        },
    )
    with pytest.raises(ConfigurationError, match="tts routing"):
        validate_runtime_configuration(configuration)


def test_coordinator_closes_profile_router_backends() -> None:
    backend = _Backend("closable")
    closed = False

    async def close() -> None:
        nonlocal closed
        closed = True

    backend.aclose = close  # type: ignore[attr-defined]
    coordinator = TTSCoordinator(TTSProfileRouter({"default": TTSProfile("default", backend)}))
    asyncio.run(coordinator.aclose())
    assert closed is True


def test_profile_router_start_initializes_each_unique_backend_once() -> None:
    class Backend:
        def __init__(self, name: str, available: bool) -> None:
            self.name = name
            self.available = available
            self.starts = 0
            self.closes = 0

        async def start(self) -> EngineHealth:
            self.starts += 1
            return EngineHealth(self.name, self.available, self.name)

        async def health(self) -> EngineHealth:
            return EngineHealth(self.name, self.available, self.name)

        async def stream(self, request: SpeechRequest):
            if False:
                yield SpeechChunk(request.request_id, b"")

        async def aclose(self) -> None:
            self.closes += 1

    shared = Backend("shared", True)
    unavailable = Backend("offline", False)
    router = TTSProfileRouter(
        {
            "zh": TTSProfile("zh", shared, languages=frozenset({"zh"})),
            "jp": TTSProfile("jp", shared, languages=frozenset({"jp"})),
            "offline": TTSProfile("offline", unavailable),
        },
        default_profile="zh",
    )

    async def scenario() -> None:
        health = await router.start()
        assert health.available
        await router.aclose()

    asyncio.run(scenario())
    assert shared.starts == 1
    assert unavailable.starts == 1
    assert shared.closes == 1
    assert unavailable.closes == 1


def test_profile_router_start_retries_unavailable_backend_on_next_start() -> None:
    class FlakyBackend:
        def __init__(self) -> None:
            self.starts = 0

        async def start(self) -> EngineHealth:
            self.starts += 1
            return EngineHealth("flaky", self.starts > 1, "flaky")

        async def stream(self, request: SpeechRequest):
            if False:
                yield SpeechChunk(request.request_id, b"")

    backend = FlakyBackend()
    router = TTSProfileRouter({"flaky": TTSProfile("flaky", backend)})

    async def scenario() -> tuple[EngineHealth, EngineHealth]:
        first = await router.start()
        second = await router.start()
        return first, second

    first, second = asyncio.run(scenario())

    assert first.available is False
    assert second.available is True
    assert backend.starts == 2


def test_profile_router_concurrent_start_is_single_flight() -> None:
    """GUI 与配置观察器并发启动时，同一模型只加载一次。"""

    class Backend:
        def __init__(self) -> None:
            self.starts = 0

        async def start(self) -> EngineHealth:
            self.starts += 1
            await asyncio.sleep(0.02)
            return EngineHealth("shared", True, "ready")

        async def stream(self, request: SpeechRequest):
            if False:
                yield SpeechChunk(request.request_id, b"")

    backend = Backend()
    router = TTSProfileRouter({"shared": TTSProfile("shared", backend)})

    async def scenario() -> None:
        first, second = await asyncio.gather(router.start(), router.start())
        assert first.available is True
        assert second.available is True

    asyncio.run(scenario())
    assert backend.starts == 1


def test_profile_router_close_waits_for_inflight_start() -> None:
    class Backend:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.events: list[str] = []

        async def start(self) -> EngineHealth:
            self.events.append("start")
            self.entered.set()
            await self.release.wait()
            return EngineHealth("race", True, "ready")

        async def stream(self, request: SpeechRequest):
            if False:
                yield SpeechChunk(request.request_id, b"")

        async def aclose(self) -> None:
            self.events.append("close")

    backend = Backend()
    router = TTSProfileRouter({"race": TTSProfile("race", backend)})

    async def scenario() -> None:
        start_task = asyncio.create_task(router.start())
        await backend.entered.wait()
        close_task = asyncio.create_task(router.aclose())
        await asyncio.sleep(0)
        assert backend.events == ["start"]
        backend.release.set()
        await asyncio.gather(start_task, close_task)

    asyncio.run(scenario())
    assert backend.events == ["start", "close"]
    assert router.diagnostics()["profiles"]


def test_profile_router_cancelled_close_can_retry_cleanup() -> None:
    class Backend:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.closes = 0

        async def stream(self, request: SpeechRequest):
            if False:
                yield SpeechChunk(request.request_id, b"")

        async def aclose(self) -> None:
            self.closes += 1
            self.entered.set()
            await self.release.wait()

    backend = Backend()
    router = TTSProfileRouter({"retry": TTSProfile("retry", backend)})

    async def scenario() -> None:
        close_task = asyncio.create_task(router.aclose())
        await backend.entered.wait()
        close_task.cancel()
        backend.release.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        assert router._closed is False
        await router.aclose()

    asyncio.run(scenario())
    assert backend.closes == 2
    assert router._closed is True


def test_clear_context_routes_preserves_pinned_route_after_flush() -> None:
    global_voice = _Backend("global")
    character_voice = _Backend("character")
    profile_router = TTSProfileRouter(
        {
            "global": TTSProfile("global", global_voice),
            "character": TTSProfile("character", character_voice),
        },
        default_profile="global",
        role_profiles={"maid": "character"},
    )
    coordinator = TTSCoordinator(profile_router)
    context = ConversationContext("profile", "local", "pinned-clear", 1)

    async def scenario() -> None:
        assert coordinator.pin_context(context, language="zh", role="maid") == (
            "zh",
            "character",
        )
        await coordinator.enqueue_text(
            context,
            "第一句。",
            language="zh",
            role="maid",
            flush=True,
        )
        coordinator.clear_context_routes()
        profile_router.reconfigure(
            {
                "global": TTSProfile("global", global_voice),
                "character": TTSProfile("character", character_voice),
            },
            default_profile="global",
            role_profiles={"maid": "global"},
        )
        await coordinator.enqueue_text(
            context,
            "第二句。",
            language="zh",
            role="maid",
            flush=True,
        )
        coordinator.release_context(context)

    asyncio.run(scenario())

    assert character_voice.calls == 2
    assert global_voice.calls == 0


def test_pinned_context_migrates_when_hot_reload_removes_its_profile() -> None:
    """移除旧 profile 后，同一回合的后续句段不能静默丢失 TTS。"""

    first_voice = _Backend("first")
    replacement_voice = _Backend("replacement")
    profile_router = TTSProfileRouter(
        {
            "first": TTSProfile("first", first_voice),
            "replacement": TTSProfile("replacement", replacement_voice),
        },
        default_profile="first",
    )
    coordinator = TTSCoordinator(profile_router)
    context = ConversationContext("profile", "local", "profile-removed", 1)

    async def scenario() -> None:
        assert coordinator.pin_context(context, language="zh") == ("zh", "first")
        await coordinator.enqueue_text(context, "旧句。", language="zh", flush=True)

        # 配置重载移除了旧 profile；当前回合仍有后续文本时，应迁移到
        # 当前可用 profile，而不是继续提交已经不存在的 profile_id。
        profile_router.reconfigure(
            {"replacement": TTSProfile("replacement", replacement_voice)},
            default_profile="replacement",
        )
        await coordinator.enqueue_text(context, "新句。", language="zh", flush=True)
        coordinator.release_context(context)

    asyncio.run(scenario())

    assert first_voice.calls == 1
    assert replacement_voice.calls == 1
