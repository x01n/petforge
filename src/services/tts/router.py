from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Any, cast

from core.tts.contracts import EngineHealth, SpeechChunk, SpeechRequest
from core.tts.language import canonical_tts_language
from logger.events import log_event

logger = logging.getLogger(__name__)


class _CrossLoopLock:
    """可跨 asyncio 事件循环等待的串行锁。

    TTS 路由器可能被配置观察器、控制台和主循环分别调用；使用
    ``asyncio.Lock`` 会把锁绑定到首次使用的事件循环，导致跨线程热重载
    直接失败。这里以线程锁为实际互斥，协程侧采用非阻塞轮询，保证取消
    不会遗留在线程池中的等待任务。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    async def __aenter__(self) -> _CrossLoopLock:
        while not self._lock.acquire(blocking=False):
            await asyncio.sleep(0.005)
        return self

    async def __aexit__(self, *_args: object) -> None:
        self._lock.release()


def _as_languages(value: object) -> frozenset[str]:
    """把 profile 的 ``languages`` 字段规范化为语言桶集合。"""

    if value is None:
        return frozenset()
    if isinstance(value, str):
        raw_values: Iterable[object] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        raw_values = value
    else:
        raise ValueError("tts profile languages must be a string or list")
    result: set[str] = set()
    for raw in raw_values:
        text = str(raw or "").strip()
        if text == "*":
            result.add("*")
            continue
        language = canonical_tts_language(text)
        if not language:
            raise ValueError("tts profile language is invalid")
        result.add(language)
    return frozenset(result)


def _as_roles(value: object) -> frozenset[str]:
    """把 profile 的 ``roles`` 字段规范化为安全角色标识集合。"""

    if value is None:
        return frozenset()
    if isinstance(value, str):
        raw_values: Iterable[object] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        raw_values = value
    else:
        raise ValueError("tts profile roles must be a string or list")
    result: set[str] = set()
    for raw in raw_values:
        role = str(raw or "").strip()
        if not role or len(role) > 128 or any(char in role for char in "\r\n\x00"):
            raise ValueError("tts profile role is invalid")
        result.add(role)
    return frozenset(result)


def _role_key(value: object) -> str:
    """返回角色路由使用的稳定键；空角色不参与自动路由。"""

    role = str(value or "").strip()
    if not role or len(role) > 128 or any(char in role for char in "\r\n\x00"):
        return ""
    return role.casefold()


def _as_bool(value: object, *, field_name: str) -> bool:
    """严格解析 profile 布尔字段，避免 ``bool("false")`` 误判。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


@dataclass(frozen=True)
class TTSProfile:
    """一个可独立熔断的 TTS 后端 profile。"""

    profile_id: str
    backend: object
    languages: frozenset[str] = frozenset()
    priority: int = 0
    cooldown_seconds: float = 30.0
    enabled: bool = True
    # 默认要求真实 PCM；纯文字/无音频后端可显式关闭，避免被错误地当作
    # 成功的空音频流，也不会阻断文本输出。
    requires_audio: bool | None = None
    # 可选角色标签；一个 profile 可以同时服务多个角色。
    roles: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id or "").strip()
        if (
            not profile_id
            or len(profile_id) > 128
            or any(char in profile_id for char in "\r\n\x00")
        ):
            raise ValueError("tts profile id is required")
        if self.backend is None or not callable(getattr(self.backend, "stream", None)):
            raise ValueError("tts profile backend must provide stream")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("tts profile priority must be an integer")
        priority = self.priority
        try:
            cooldown = float(self.cooldown_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("tts profile cooldown_seconds must be a number") from exc
        if not isfinite(cooldown) or cooldown < 0:
            raise ValueError("tts profile cooldown_seconds must be non-negative and finite")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "languages", _as_languages(self.languages))
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "cooldown_seconds", cooldown)
        object.__setattr__(
            self,
            "enabled",
            _as_bool(self.enabled, field_name="tts profile enabled"),
        )
        requires_audio = self.requires_audio
        if requires_audio is None:
            requires_audio = getattr(
                self.backend,
                "requires_audio",
                getattr(self.backend, "require_audio", True),
            )
        object.__setattr__(
            self,
            "requires_audio",
            _as_bool(requires_audio, field_name="tts profile requires_audio"),
        )
        object.__setattr__(self, "roles", _as_roles(self.roles))

    @classmethod
    def from_mapping(cls, profile_id: str, backend: object, value: Mapping[str, Any]) -> TTSProfile:
        """从已经构造好的后端和 YAML profile 字段创建 profile。"""

        if not isinstance(value, Mapping):
            raise ValueError(f"tts.profiles.{profile_id} must be a mapping")
        return cls(
            profile_id=profile_id,
            backend=backend,
            languages=_as_languages(value.get("languages")),
            priority=value.get("priority", 0),
            cooldown_seconds=value.get("cooldown_seconds", 30.0),
            enabled=value.get("enabled", True),
            requires_audio=value.get(
                "requires_audio",
                value.get(
                    "require_audio",
                    getattr(backend, "requires_audio", getattr(backend, "require_audio", True)),
                ),
            ),
            roles=value.get("roles", ()),
        )


@dataclass
class _FailureState:
    failures: int = 0
    cooldown_until: float = 0.0
    last_error: str = ""


class TTSProfileRouter:
    """实现 ``SpeechBackend`` 协议的 profile 路由器。

    路由优先级依次为显式 profile、角色映射与角色标签、全局选择、语言映射、
    profile 声明的语言、默认 profile、回退列表和剩余 profile。一次请求只有在后端尚未产生音频时才会
    尝试下一个后端；已经产生部分音频后继续回退会造成重复播放，因此直接
    抛出原错误。失败 profile 使用指数冷却，成功完成一整段流后清除失败状态。
    """

    def __init__(
        self,
        profiles: Mapping[str, TTSProfile | object] | Sequence[TTSProfile] = (),
        *,
        default_profile: str = "",
        language_profiles: Mapping[str, str] | None = None,
        fallback_profiles: Sequence[str] = (),
        role_profiles: Mapping[str, str] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._clock = clock or monotonic
        self._lock = threading.RLock()
        self._profiles: dict[str, TTSProfile] = {}
        self._language_profiles: dict[str, str] = {}
        self._role_profiles: dict[str, str] = {}
        self._fallback_profiles: tuple[str, ...] = ()
        self._default_profile = ""
        self._selected_profile = ""
        self._failures: dict[str, _FailureState] = {}
        # 后端对象按身份记录启动代次；同一对象被多个 profile 复用或重复
        # 调用 start() 时只初始化一次，热重载替换对象后再启动新代次。
        self._started_backend_ids: set[int] = set()
        self._backend_health: dict[int, EngineHealth] = {}
        # 多个事件循环可能同时触发启动（例如 GUI 与配置观察器）；
        # 必须保证同一后端只执行一次模型初始化。
        self._start_lock = _CrossLoopLock()
        self._closing = False
        self._closed = False
        self.reconfigure(
            profiles,
            default_profile=default_profile,
            language_profiles=language_profiles,
            fallback_profiles=fallback_profiles,
            role_profiles=role_profiles,
        )

    @property
    def backend(self) -> object:
        """兼容旧调用方读取当前后端对象。"""

        with self._lock:
            profile_id = self._selected_profile or self._default_profile
            if not profile_id and self._profiles:
                profile_id = next(iter(self._profiles))
            profile = self._profiles.get(profile_id)
            if profile is None:
                raise RuntimeError("no TTS profile is configured")
            return profile.backend

    @property
    def active_profile(self) -> str:
        """返回显式选择、默认选择或优先级最高的 profile。"""

        with self._lock:
            return self._active_profile_id_locked()

    @property
    def selected_profile(self) -> str:
        """返回显式选择；为空时表示使用自动路由。"""

        with self._lock:
            return self._selected_profile

    def profile_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._profiles)

    def profile_enabled(self, profile_id: str) -> bool:
        """返回精确 profile 是否存在且已启用。

        协调器在固定一个回合的音色前使用这个只读探针，避免把已禁用的
        profile 写入回合快照后再静默回退到其它音色。
        """

        target = str(profile_id or "").strip()
        with self._lock:
            profile = self._profiles.get(target)
            return bool(profile is not None and profile.enabled)

    def _active_profile_id_locked(self) -> str:
        explicit = self._selected_profile
        if explicit in self._profiles and self._profiles[explicit].enabled:
            return explicit
        if (
            self._default_profile in self._profiles
            and self._profiles[self._default_profile].enabled
        ):
            return self._default_profile
        enabled = [profile for profile in self._profiles.values() if profile.enabled]
        if not enabled:
            return ""
        return max(enabled, key=lambda item: (item.priority, item.profile_id)).profile_id

    def select(self, profile_id: str | None = None, *, language: str | None = None) -> str:
        """运行时选择 profile，或为指定语言更新映射。

        ``select(None)`` 清除全局显式选择并恢复默认路由。语言映射只影响后续
        请求，不会中断已经建立的音频流。
        """

        with self._lock:
            if language is not None:
                normalized_language = canonical_tts_language(language)
                if not normalized_language:
                    raise ValueError("tts profile language is invalid")
                target = str(profile_id or "").strip()
                if target and target not in self._profiles:
                    raise KeyError(f"unknown TTS profile: {target}")
                if target and not self._profiles[target].enabled:
                    raise ValueError(f"TTS profile is disabled: {target}")
                if target:
                    self._language_profiles[normalized_language] = target
                else:
                    self._language_profiles.pop(normalized_language, None)
                return target or self._active_profile_id_locked()
            target = str(profile_id or "").strip()
            if target and target not in self._profiles:
                raise KeyError(f"unknown TTS profile: {target}")
            if target and not self._profiles[target].enabled:
                raise ValueError(f"TTS profile is disabled: {target}")
            self._selected_profile = target
            return self._active_profile_id_locked()

    def select_profile(self, profile_id: str | None = None, *, language: str | None = None) -> str:
        """``select`` 的显式命名别名，便于控制面调用。"""

        return self.select(profile_id, language=language)

    def reconfigure(
        self,
        profiles: Mapping[str, TTSProfile | object] | Sequence[TTSProfile],
        *,
        default_profile: str = "",
        language_profiles: Mapping[str, str] | None = None,
        fallback_profiles: Sequence[str] = (),
        role_profiles: Mapping[str, str] | None = None,
    ) -> tuple[object, ...]:
        """原子替换 profile 表并返回不再使用的旧后端。

        返回的后端不会在此处关闭，调用方可以在确认没有活动流后异步回收，
        避免热切换时中断旧请求。
        """

        normalized_profiles: dict[str, TTSProfile] = {}
        if isinstance(profiles, Mapping):
            entries = profiles.items()
        else:
            entries = ((item.profile_id, item) for item in profiles)
        for raw_id, raw_profile in entries:
            profile_id = str(raw_id or "").strip()
            if isinstance(raw_profile, TTSProfile):
                profile = raw_profile
                if profile.profile_id != profile_id:
                    profile = TTSProfile(
                        profile_id=profile_id,
                        backend=profile.backend,
                        languages=profile.languages,
                        priority=profile.priority,
                        cooldown_seconds=profile.cooldown_seconds,
                        enabled=profile.enabled,
                        requires_audio=profile.requires_audio,
                        roles=profile.roles,
                    )
            else:
                profile = TTSProfile(profile_id=profile_id, backend=raw_profile)
            if profile_id in normalized_profiles:
                raise ValueError(f"duplicate TTS profile id: {profile_id}")
            normalized_profiles[profile_id] = profile

        default = str(default_profile or "").strip()
        if default and default not in normalized_profiles:
            raise KeyError(f"unknown default TTS profile: {default}")
        if language_profiles is not None and not isinstance(language_profiles, Mapping):
            raise ValueError("tts language_profiles must be a mapping")
        normalized_language_profiles: dict[str, str] = {}
        for raw_language, raw_profile_id in (language_profiles or {}).items():
            language = canonical_tts_language(raw_language)
            profile_id = str(raw_profile_id or "").strip()
            if not language:
                raise ValueError("tts language_profiles contains an invalid language")
            if not profile_id or profile_id not in normalized_profiles:
                raise KeyError(f"unknown TTS profile for language: {language}")
            if not normalized_profiles[profile_id].enabled:
                raise ValueError(f"TTS profile is disabled: {profile_id}")
            normalized_language_profiles[language] = profile_id
        normalized_role_profiles: dict[str, str] = {}
        if role_profiles is not None and not isinstance(role_profiles, Mapping):
            raise ValueError("tts routing.role_profiles must be a mapping")
        for raw_role, raw_profile_id in (role_profiles or {}).items():
            role = str(raw_role or "").strip()
            role_key = _role_key(role)
            profile_id = str(raw_profile_id or "").strip()
            if not role_key or role_key == "*":
                raise ValueError("tts routing.role_profiles contains an invalid role")
            if not profile_id or profile_id not in normalized_profiles:
                raise KeyError(f"unknown TTS profile for role: {role}")
            if not normalized_profiles[profile_id].enabled:
                raise ValueError(f"TTS profile is disabled: {profile_id}")
            normalized_role_profiles[role_key] = profile_id

        normalized_fallbacks: list[str] = []
        if isinstance(fallback_profiles, (str, bytes, bytearray)) or not isinstance(
            fallback_profiles, Iterable
        ):
            raise ValueError("tts fallback_profiles must be a list")
        for raw_profile_id in fallback_profiles:
            profile_id = str(raw_profile_id or "").strip()
            if not profile_id or profile_id not in normalized_profiles:
                raise KeyError(f"unknown fallback TTS profile: {profile_id}")
            if not normalized_profiles[profile_id].enabled:
                raise ValueError(f"TTS fallback profile is disabled: {profile_id}")
            if profile_id not in normalized_fallbacks:
                normalized_fallbacks.append(profile_id)

        with self._lock:
            if self._closed:
                raise RuntimeError("TTS profile router is closed")
            old_profiles = self._profiles
            old_backends = tuple(profile.backend for profile in old_profiles.values())
            self._profiles = normalized_profiles
            self._default_profile = default
            self._language_profiles = normalized_language_profiles
            self._role_profiles = normalized_role_profiles
            self._fallback_profiles = tuple(normalized_fallbacks)
            if self._selected_profile not in normalized_profiles:
                self._selected_profile = ""
            # 同一 profile 且后端对象未更换时保留冷却，避免热重载瞬间绕过熔断。
            self._failures = {
                profile_id: self._failures[profile_id]
                for profile_id in normalized_profiles
                if profile_id in self._failures
                and profile_id in old_profiles
                and old_profiles[profile_id].backend is normalized_profiles[profile_id].backend
            }
            new_backends = tuple(profile.backend for profile in normalized_profiles.values())
            active_backend_ids = {id(backend) for backend in new_backends}
            self._started_backend_ids.intersection_update(active_backend_ids)
            self._backend_health = {
                backend_id: health
                for backend_id, health in self._backend_health.items()
                if backend_id in active_backend_ids
            }
            retired: list[object] = []
            for index, backend in enumerate(old_backends):
                if backend is None or any(backend is current for current in new_backends):
                    continue
                if any(backend is prior for prior in old_backends[:index]):
                    continue
                retired.append(backend)
            return tuple(retired)

    def reconfigure_profiles(
        self,
        profiles: Mapping[str, TTSProfile | object] | Sequence[TTSProfile],
        *,
        default_profile: str = "",
        language_profiles: Mapping[str, str] | None = None,
        fallback_profiles: Sequence[str] = (),
        role_profiles: Mapping[str, str] | None = None,
    ) -> tuple[object, ...]:
        """``reconfigure`` 的显式命名别名。"""

        return self.reconfigure(
            profiles,
            default_profile=default_profile,
            language_profiles=language_profiles,
            fallback_profiles=fallback_profiles,
            role_profiles=role_profiles,
        )

    def diagnostics(self) -> dict[str, object]:
        """返回不含后端配置细节的路由状态。"""

        now = self._clock()
        with self._lock:
            profiles = tuple(
                {
                    "id": profile.profile_id,
                    "languages": tuple(sorted(profile.languages)),
                    "roles": tuple(sorted(profile.roles)),
                    "priority": profile.priority,
                    "enabled": profile.enabled,
                    "requires_audio": profile.requires_audio,
                    "available": bool(
                        self._backend_health.get(
                            id(profile.backend), EngineHealth("", False)
                        ).available
                    ),
                    "failures": self._failures.get(profile.profile_id, _FailureState()).failures,
                    "cooldown_remaining": max(
                        0.0,
                        self._failures.get(profile.profile_id, _FailureState()).cooldown_until
                        - now,
                    ),
                    "last_error": self._failures.get(
                        profile.profile_id, _FailureState()
                    ).last_error,
                }
                for profile in self._profiles.values()
            )
            return {
                "active_profile": self._active_profile_id_locked(),
                "default_profile": self._default_profile,
                "language_profiles": dict(self._language_profiles),
                "role_profiles": dict(self._role_profiles),
                "fallback_profiles": self._fallback_profiles,
                "profiles": profiles,
            }

    def _candidate_ids(
        self,
        language: str,
        *,
        preferred_profile: str = "",
        role: str = "",
    ) -> tuple[str, ...]:
        normalized_language = canonical_tts_language(language)
        preferred = str(preferred_profile or "").strip()
        with self._lock:
            ordered: list[str] = []

            def add(profile_id: str) -> None:
                if profile_id and profile_id in self._profiles and profile_id not in ordered:
                    if self._profiles[profile_id].enabled:
                        ordered.append(profile_id)

            # 显式 voice 优先；指定角色时角色路由优先于全局选择，
            # 避免“当前全局音色”覆盖模型/工具明确要求的角色。
            add(preferred)
            role_key = _role_key(role)
            if role_key:
                add(self._role_profiles.get(role_key, ""))
                for profile in self._profiles.values():
                    if profile.enabled and any(
                        _role_key(item) == role_key for item in profile.roles
                    ):
                        add(profile.profile_id)
            add(self._selected_profile)
            add(self._language_profiles.get(normalized_language, ""))
            language_profiles = sorted(
                (
                    profile
                    for profile in self._profiles.values()
                    if profile.enabled
                    and ("*" in profile.languages or normalized_language in profile.languages)
                ),
                key=lambda item: (-item.priority, item.profile_id),
            )
            for profile in language_profiles:
                add(profile.profile_id)
            add(self._default_profile)
            for profile_id in self._fallback_profiles:
                add(profile_id)
            for profile in sorted(
                (item for item in self._profiles.values() if item.enabled),
                key=lambda item: (-item.priority, item.profile_id),
            ):
                add(profile.profile_id)
            if not ordered:
                return ()
            now = self._clock()
            available = [
                profile_id
                for profile_id in ordered
                if self._failures.get(profile_id, _FailureState()).cooldown_until <= now
            ]
            # 所有 profile 都在冷却时也要按既定顺序尝试完整回退链。只尝试
            # 首个 profile 会让一次短暂的共同故障把后备模型永久挡住：首个
            # profile 失败后 ``stream`` 无法再看到其它 profile，直到冷却期
            # 结束才会恢复。完整重试仍受本次请求的 profile 列表限制，不会
            # 引入未配置的后端；下一个成功 profile 会清除自身失败状态。
            return tuple(available or ordered)

    def profile_for(self, language: str = "zh", *, role: str = "") -> str:
        """返回当前语言和角色在本次请求中会优先使用的 profile 标识。"""

        profiles = self._candidate_ids(language, role=role)
        return profiles[0] if profiles else ""

    def _record_failure(
        self, profile_id: str, error: BaseException, *, backend: object | None = None
    ) -> None:
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None or (backend is not None and profile.backend is not backend):
                return
            state = self._failures.setdefault(profile_id, _FailureState())
            state.failures += 1
            state.last_error = type(error).__name__
            multiplier = 2 ** min(state.failures - 1, 5)
            state.cooldown_until = self._clock() + min(profile.cooldown_seconds * multiplier, 300.0)

    def _record_success(self, profile_id: str, *, backend: object | None = None) -> None:
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is not None and (backend is None or profile.backend is backend):
                self._failures.pop(profile_id, None)

    async def start(self) -> EngineHealth:
        """串行初始化全部启用 profile，避免并发重复加载模型。"""

        async with self._start_lock:
            return await self._start_once()

    async def _start_once(self) -> EngineHealth:
        """并行初始化全部启用且去重后的 profile 后端。

        同一后端对象可能被多个 profile 共享；启动代次按对象身份缓存，
        重复调用不会再次触发外部 worker/模型加载。配置热重载替换对象
        后，只有新对象进入下一次初始化。
        """

        with self._lock:
            if self._closed:
                return EngineHealth("tts-router", False, "TTS profile router is closed")
            backends: list[tuple[str, object]] = []
            for profile in self._profiles.values():
                if not profile.enabled:
                    continue
                if any(profile.backend is backend for _, backend in backends):
                    continue
                backends.append((profile.profile_id, profile.backend))
            pending = [
                (profile_id, backend)
                for profile_id, backend in backends
                if id(backend) not in self._started_backend_ids
            ]
            cached_health = tuple(
                self._backend_health.get(id(backend))
                for _, backend in backends
                if id(backend) in self._backend_health
            )
        if not backends:
            return EngineHealth("tts-router", False, "no TTS profile is configured")

        async def initialize(profile_id: str, backend: object) -> EngineHealth:
            initializer = getattr(backend, "start", None)
            try:
                if callable(initializer):
                    result = initializer()
                else:
                    health_method = getattr(backend, "health", None)
                    if not callable(health_method):
                        return EngineHealth(
                            f"tts-router:{profile_id}",
                            False,
                            "backend does not provide health",
                        )
                    result = health_method()
                result = await result if inspect.isawaitable(result) else result
                if isinstance(result, EngineHealth):
                    return result
                return EngineHealth(
                    f"tts-router:{profile_id}",
                    bool(result),
                    "backend initialization completed",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    logger,
                    "tts.profile.initialize",
                    component="tts.router",
                    status="failed",
                    level=logging.WARNING,
                    fields={"profile": profile_id, "error": type(exc).__name__},
                )
                return EngineHealth(
                    f"tts-router:{profile_id}",
                    False,
                    f"backend initialization failed: {type(exc).__name__}",
                )

        if pending:
            results = await asyncio.gather(
                *(initialize(profile_id, backend) for profile_id, backend in pending)
            )
            with self._lock:
                for (_, backend), result in zip(pending, results, strict=True):
                    self._backend_health[id(backend)] = result
                    # 不可用通常是外部服务暂时未启动或模型仍在准备；
                    # 不把失败结果锁成“已启动”，下一次 start 才能重试。
                    if result.available:
                        self._started_backend_ids.add(id(backend))
                cached_health = tuple(
                    self._backend_health.get(id(backend))
                    for _, backend in backends
                    if id(backend) in self._backend_health
                )
        available = sum(1 for result in cached_health if result is not None and result.available)
        log_event(
            logger,
            "tts.router.initialize",
            component="tts.router",
            status="ready" if available else "unavailable",
            level=logging.INFO if available else logging.WARNING,
            fields={"available": available, "total": len(backends)},
        )
        return EngineHealth(
            "tts-router",
            available > 0,
            f"{available}/{len(backends)} TTS profiles are available",
        )

    async def health(self, *, language: str = "zh", profile: str | None = None) -> EngineHealth:
        """探测指定语言的路由候选，首个可用 profile 即视为路由可用。"""

        if profile is not None:
            profile_id = str(profile or "").strip()
            with self._lock:
                known = profile_id in self._profiles and self._profiles[profile_id].enabled
            candidates = (profile_id,) if known else ()
        else:
            candidates = self._candidate_ids(language)
        if not candidates:
            return EngineHealth("tts-router", False, "no TTS profile is configured")
        failures: list[str] = []
        for profile_id in candidates:
            with self._lock:
                profile_config = self._profiles.get(profile_id)
            if profile_config is None:
                continue
            try:
                health_method = getattr(profile_config.backend, "health", None)
                if not callable(health_method):
                    raise RuntimeError("TTS profile backend does not provide health")
                result = cast(Callable[[], object], health_method)()
                result = await result if inspect.isawaitable(result) else result
            except Exception as exc:
                failures.append(f"{profile_id}:{type(exc).__name__}")
                with self._lock:
                    self._backend_health[id(profile_config.backend)] = EngineHealth(
                        f"tts-router:{profile_id}", False, type(exc).__name__
                    )
                continue
            if isinstance(result, EngineHealth):
                with self._lock:
                    self._backend_health[id(profile_config.backend)] = result
                if result.available:
                    return EngineHealth(
                        f"tts-router:{profile_id}", True, result.message, result.latency_ms
                    )
                failures.append(f"{profile_id}:unavailable")
            elif result:
                normalized_health = EngineHealth(
                    f"tts-router:{profile_id}", True, "backend is available"
                )
                with self._lock:
                    self._backend_health[id(profile_config.backend)] = normalized_health
                return normalized_health
            else:
                normalized_health = EngineHealth(
                    f"tts-router:{profile_id}", False, "backend is unavailable"
                )
                with self._lock:
                    self._backend_health[id(profile_config.backend)] = normalized_health
                failures.append(f"{profile_id}:unavailable")
        return EngineHealth(
            "tts-router",
            False,
            "no TTS profile is available" + (f" ({','.join(failures)})" if failures else ""),
        )

    async def health_for(self, language: str = "zh", profile: str | None = None) -> EngineHealth:
        """``health`` 的位置参数别名，便于状态面调用。"""

        return await self.health(language=language, profile=profile)

    async def stream(self, request: SpeechRequest) -> AsyncIterator[SpeechChunk]:
        with self._lock:
            closed = self._closed or self._closing
        if closed:
            raise RuntimeError("TTS profile router is closed")
        preferred_profile = str(getattr(request, "profile_id", "") or "").strip()
        request_role = str(getattr(request, "role", "") or "").strip()
        if preferred_profile:
            # ``profile_id`` 是调用方明确指定的音色；若它被禁用或已在热
            # 重载中移除，静默改用全局音色会让角色声音不可预测。协调器
            # 通常会提前检查，这里仍在路由边界再次拒绝直接调用。
            with self._lock:
                preferred = self._profiles.get(preferred_profile)
            if preferred is None or not preferred.enabled:
                raise RuntimeError("requested TTS profile is unavailable")
        profiles = self._candidate_ids(
            request.language,
            preferred_profile=preferred_profile,
            role=request_role,
        )
        if not profiles:
            raise RuntimeError("no TTS profile is configured")
        last_error: BaseException | None = None
        for profile_id in profiles:
            with self._lock:
                profile = self._profiles.get(profile_id)
            if profile is None or not profile.enabled:
                continue
            stream: AsyncIterator[SpeechChunk] | None = None
            yielded_audio = False
            completed = False
            pending_empty: list[SpeechChunk] = []
            try:
                stream_method = getattr(profile.backend, "stream", None)
                if not callable(stream_method):
                    raise RuntimeError("TTS profile backend stream is not asynchronous")
                stream_value = cast(Callable[[SpeechRequest], object], stream_method)(request)
                if not hasattr(stream_value, "__aiter__"):
                    raise RuntimeError("TTS profile backend stream is not asynchronous")
                stream = cast(AsyncIterator[SpeechChunk], stream_value)
                async for chunk in stream:
                    chunk_request_id = str(getattr(chunk, "request_id", "") or "").strip()
                    if chunk_request_id != request.request_id:
                        raise RuntimeError("TTS profile returned mismatched request_id")
                    data = getattr(chunk, "data", b"")
                    if not data:
                        # 先暂存空终态；如果这个 profile 后续没有任何
                        # PCM，则必须在选择后备 profile 时丢弃它。
                        pending_empty.append(chunk)
                        continue
                    if pending_empty:
                        for empty_chunk in pending_empty:
                            yield empty_chunk
                        pending_empty.clear()
                    yielded_audio = True
                    yield chunk
                if profile.requires_audio and not yielded_audio:
                    # 空流或只有空的终止标记不是成功的音频回退；否则
                    # TextOnly 之后的真实后端会被错误地标记为健康。
                    raise RuntimeError("TTS profile ended without audio data")
                if not yielded_audio:
                    # 只有明确允许无音频的后端（例如 TextOnly）才能保留
                    # 空终态；其余空块在切换后备 profile 时不可见。
                    for empty_chunk in pending_empty:
                        yield empty_chunk
                completed = True
                self._record_success(profile_id, backend=profile.backend)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_failure(profile_id, exc, backend=profile.backend)
                last_error = exc
                if yielded_audio:
                    raise
            finally:
                if stream is not None and not completed:
                    close_stream = getattr(stream, "aclose", None)
                    if callable(close_stream):
                        try:
                            result = close_stream()
                            if inspect.isawaitable(result):
                                await result
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            pass
        if last_error is not None:
            raise RuntimeError("all TTS profiles failed") from last_error
        raise RuntimeError("no enabled TTS profile is available")

    async def aclose(self) -> None:
        """关闭路由器及当前 profile 后端（去重）。"""

        # 关闭必须与启动共享同一 single-flight 锁。否则配置重载恰好在
        # ``start`` 初始化模型时调用 ``aclose``，关闭流程可能先返回，随后
        # 启动协程又把已关闭的 worker 留在运行状态。
        async with self._start_lock:
            await self._aclose_once()

    async def _aclose_once(self) -> None:
        """在启动锁内执行一次真实后端清理。"""

        with self._lock:
            if self._closed:
                return
            self._closing = True
            self._closed = True
            backends: list[object] = []
            for profile in self._profiles.values():
                if not any(profile.backend is existing for existing in backends):
                    backends.append(profile.backend)
        try:
            for backend in backends:
                close = getattr(backend, "aclose", None)
                if not callable(close):
                    close = getattr(backend, "close", None)
                if not callable(close):
                    continue
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log_event(
                        logger,
                        "tts.profile.unload",
                        component="tts.router",
                        status="failed",
                        level=logging.WARNING,
                        fields={"error": type(exc).__name__},
                    )
        except asyncio.CancelledError:
            # 关闭等待者可被取消，但不能把尚未清理完的后端永久标成
            # closed；下一次 aclose 必须能够继续补偿清理。
            with self._lock:
                self._closed = False
                self._closing = False
            raise
        log_event(
            logger,
            "tts.router.unload",
            component="tts.router",
            status="completed",
            fields={"backends": len(backends)},
        )
        with self._lock:
            self._closed = True
            self._closing = False


TTSBackendRouter = TTSProfileRouter

__all__ = ["TTSBackendRouter", "TTSProfile", "TTSProfileRouter"]
