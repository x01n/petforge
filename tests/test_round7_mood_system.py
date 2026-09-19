"""第 7 轮「脾气系统」验收测试。

覆盖：9 值词汇集统一、set/get 归一、陈旧心情衰减提示、旧构造零回归、
proactive 心情系数采样、对话提示注入、公开投影白名单与 directives 同源。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.loader import ConfigurationError, LoadedConfiguration
from db import Database
from gui.web.control_surface import public_control_state
from services.affection import MOOD_ALIASES, MOODS, AffectionService
from services.conversation import ConversationService
from services.directives import _PET_MOODS, mood_description, validate_directives
from services.proactive import ProactiveCoordinator, ProactiveSettings
from services.proactive.coordinator import ProactiveEvent

AUTHORITATIVE_MOODS = ("高兴", "平静", "困倦", "烦躁", "难过", "期待", "好奇", "生气", "孤独")


class _Memory:
    """proactive 替身：零输出记忆上下文。"""

    def build_context_prompt(self, query: str, *, max_chars: int) -> str:
        del query, max_chars
        return ""


class _Affection:
    """proactive 替身：固定好感度。"""

    def get_affection(self) -> int:
        return 42


class _Conversation:
    """proactive 替身：记录调用，立即完成。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.active = False
        self.pending: list[object] = []

    def has_active_conversation(self) -> bool:
        return self.active or bool(self.pending)

    async def complete(self, prompt: str, **_kwargs: object) -> object:
        self.calls.append(prompt)
        return SimpleNamespace(status="completed")

    def pending_approvals_for_ui(self) -> tuple[object, ...]:
        return ()


def _settings(**overrides: object) -> ProactiveSettings:
    values: dict[str, object] = {
        "enabled": True,
        "hourly_budget": 6,
        "daily_budget": 24,
        "global_cooldown_seconds": 0,
        "dedupe_seconds": 0,
        "rules": [{"id": "basic", "event": "window_active", "instruction": "关心用户"}],
    }
    values.update(overrides)
    return ProactiveSettings.from_mapping(values)


def _stale_service(tmp_path: Path, *, hours_ago: float) -> AffectionService:
    import time

    database = Database(tmp_path / "mood.sqlite3")
    service = AffectionService(database, decay_after_seconds=14_400.0)
    timestamp = time.time() - hours_ago * 3600.0
    with database._lock:
        database.connection.execute(
            "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)", ("mood", "烦躁")
        )
        database.connection.execute(
            "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
            ("mood_updated", str(timestamp)),
        )
        database.connection.commit()
    return service


def test_mood_vocabulary_matches_directives_and_is_nine_values() -> None:
    assert set(MOODS) == set(AUTHORITATIVE_MOODS)
    assert len(MOODS) == 9
    assert set(_PET_MOODS) == set(MOODS)
    assert MOOD_ALIASES == {"开心": "高兴", "忧郁": "难过"}


def test_set_mood_normalizes_legacy_values_and_get_reads_them(tmp_path: Path) -> None:
    service = AffectionService(Database(tmp_path / "set.sqlite3"))
    assert service.set_mood("开心") == "高兴"
    assert service.get_mood() == "高兴"
    service.set_mood("忧郁")
    assert service.get_mood() == "难过"
    for mood in AUTHORITATIVE_MOODS:
        assert service.set_mood(mood) == mood
        assert service.get_mood() == mood


def test_set_mood_rejects_unknown_values_and_resets_are_untouched(tmp_path: Path) -> None:
    service = AffectionService(Database(tmp_path / "reject.sqlite3"))
    with pytest.raises(ValueError):
        service.set_mood("暴怒")
    with pytest.raises(ValueError):
        service.set_mood("happy")
    assert service.get_mood() == "平静"


def test_get_mood_falls_back_to_calm_for_missing_or_legacy_rows(tmp_path: Path) -> None:
    database = Database(tmp_path / "fallback.sqlite3")
    service = AffectionService(database)
    assert service.get_mood() == "平静"
    with database._lock:
        database.connection.execute(
            "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)", ("mood", "开心")
        )
        database.connection.commit()
    assert service.get_mood() == "高兴"
    with database._lock:
        database.connection.execute(
            "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)", ("mood", "狂怒")
        )
        database.connection.commit()
    assert service.get_mood() == "平静"


def test_stale_mood_hint_gates_on_threshold_probability_switch_and_mood(
    tmp_path: Path,
) -> None:
    service = _stale_service(tmp_path, hours_ago=5)
    # 概率未中签（注入 rng 返回 1.0 恒不命中 0.5）→ 空串
    assert service.stale_mood_hint(rng=lambda: 1.0) == ""
    # 中签（rng 0.0 < 0.5）→ 提示包含关键语义
    hint = service.stale_mood_hint(rng=lambda: 0.0)
    assert "不强制" in hint and "<mood>" in hint and "meapet" in hint
    assert "5 小时" in hint
    # 未过阈值的服务不参与判定
    fresh = AffectionService(Database(tmp_path / "fresh.sqlite3"), decay_after_seconds=14_400.0)
    import time

    with fresh._database._lock:
        fresh._database.connection.execute(
            "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)", ("mood", "烦躁")
        )
        fresh._database.connection.execute(
            "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
            ("mood_updated", str(time.time() - 60)),
        )
        fresh._database.connection.commit()
    assert fresh.stale_mood_hint(rng=lambda: 0.0) == ""
    # 平静心情即使逾期也不提示
    calm = AffectionService(Database(tmp_path / "calm.sqlite3"), decay_after_seconds=600.0)
    with calm._database._lock:
        calm._database.connection.execute(
            "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)", ("mood", "平静")
        )
        calm._database.connection.execute(
            "INSERT OR REPLACE INTO mea_state (key, value) VALUES (?, ?)",
            ("mood_updated", str(time.time() - 9999)),
        )
        calm._database.connection.commit()
    assert calm.stale_mood_hint(rng=lambda: 0.0) == ""
    # 开关关闭直接返回空串
    service.configure_mood(
        decay_enabled=False, decay_after_seconds=600.0, decay_prompt_probability=0.5
    )
    assert service.stale_mood_hint(rng=lambda: 0.0) == ""


def test_legacy_affection_constructor_stays_behavior_compatible(tmp_path: Path) -> None:
    service = AffectionService(Database(tmp_path / "legacy.sqlite3"))
    assert service.get() == 5
    assert service.set_mood("高兴") == "高兴"
    assert service.get_mood() == "高兴"
    assert service.stale_mood_hint(rng=lambda: 0.0) == ""  # 默认语义下默认构造可调用


def test_proactive_mood_multipliers_defaults_and_validation() -> None:
    settings = ProactiveSettings()
    assert settings.mood_gate_enabled is True
    assert settings.mood_multipliers == {
        "烦躁": 0.5,
        "生气": 0.5,
        "难过": 0.5,
        "高兴": 1.2,
        "期待": 1.2,
        "好奇": 1.2,
    }
    with pytest.raises(ValueError):
        ProactiveSettings.from_mapping({"mood_multipliers": {"暴怒": 0.5}})
    with pytest.raises(ValueError):
        ProactiveSettings.from_mapping({"mood_multipliers": {"烦躁": 3.0}})
    with pytest.raises(ValueError):
        ProactiveSettings.from_mapping({"mood_multipliers": []})
    with pytest.raises(ValueError):
        ProactiveSettings.from_mapping({"mood_gate_enabled": "yes"})


def test_proactive_mood_sampling_uses_injected_rng() -> None:
    class Roller:
        def __init__(self, value: float) -> None:
            self.value = value
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return self.value

    async def run(roll: float, expect_status: str) -> None:
        settings = _settings(mood_multipliers={"烦躁": 0.5})
        rolls = Roller(roll)
        coordinator = ProactiveCoordinator(
            _Conversation(),
            memory=_Memory(),
            affection=_Affection(),
            settings=settings,
            activity_provider=lambda: True,
            gate_provider=lambda: {},
            mood_provider=lambda: "烦躁",
            rng=rolls,
        )
        result = await coordinator.notify("window_active")
        assert result["status"] == expect_status, result
        assert rolls.calls >= 1
        if expect_status == "skipped_mood":
            # 跳过不写去重、不扣预算：既无入账也没有已验收计数
            assert coordinator._dedupe == {}
            assert coordinator.diagnostics()["accepted"] == 0
        await coordinator.close()

    # f=0.5：rng < f 继续，rng >= f 跳过（阈值取等号语义）
    asyncio.run(run(0.4, "queued"))
    asyncio.run(run(0.49, "queued"))
    asyncio.run(run(0.5, "skipped_mood"))
    asyncio.run(run(1.0, "skipped_mood"))


def test_proactive_mood_gate_disabled_skips_sampling() -> None:
    async def run() -> None:
        settings = _settings(mood_gate_enabled=False, mood_multipliers={"烦躁": 0.5})
        coordinator = ProactiveCoordinator(
            _Conversation(),
            memory=_Memory(),
            affection=_Affection(),
            settings=settings,
            activity_provider=lambda: True,
            gate_provider=lambda: {},
            mood_provider=lambda: "烦躁",
            rng=lambda: 1.0,
        )
        result = await coordinator.notify("window_active")
        assert result["status"] == "queued", result
        await coordinator.close()

    asyncio.run(run())


def test_event_prompt_includes_current_mood(tmp_path: Path) -> None:
    del tmp_path
    settings = _settings()
    rule = next(iter(settings.rules))
    coordinator = ProactiveCoordinator(
        _Conversation(),
        memory=_Memory(),
        affection=_Affection(),
        settings=settings,
        mood_provider=lambda: "期待",
    )
    event = ProactiveEvent(
        event_name="window_active",
        rule=rule,
        payload={},
        fingerprint="fp",
        observed_at=1.0,
    )
    prompt = coordinator._event_prompt(event)
    assert "当前心情" in prompt
    assert "期待" in prompt
    assert "兴奋" in prompt  # mood_description 短语同现


def test_conversation_build_messages_injects_mood_hint() -> None:
    from core.events.types import ConversationContext
    from services.model_routing import ModelRouter

    context = ConversationContext(profile_id="p", session_id="s", turn_id="t", generation_id=1)
    service = ConversationService(
        ModelRouter(),
        mood_hint_provider=lambda: "距上次心情标记已超过 5 小时；请判断是否更新 <mood>。",
    )
    messages = asyncio.run(service._build_messages(context, "你好", ()))
    system_texts = [
        message.content
        for message in messages
        if message.role == "system" and isinstance(message.content, str)
    ]
    assert any("<mood>" in text and "5 小时" in text for text in system_texts)
    # 独立 system 部分：存在一个不含人设/语言前缀的提示 part
    assert any("判断是否更新" in text for text in system_texts)
    # 空提示不产生额外 system 部分
    service_without = ConversationService(
        ModelRouter(),
        mood_hint_provider=lambda: "",
    )
    messages_without = asyncio.run(service_without._build_messages(context, "你好", ()))
    assert all("<mood>" not in str(message.content) for message in messages_without)


def test_web_affection_projection_includes_mood_whitelist() -> None:
    state = public_control_state({"affection": {"mood": "高兴"}})
    assert state["affection"]["mood"] == "高兴"
    state = public_control_state({"affection": {"mood": "旧秘密情绪"}})
    assert state["affection"]["mood"] == ""
    state = public_control_state({"affection": {"mood": 1, "current": 50}})
    assert state["affection"]["mood"] == ""
    for mood in AUTHORITATIVE_MOODS:
        assert public_control_state({"affection": {"mood": mood}})["affection"]["mood"] == mood


def test_directives_pet_moods_same_source_and_validate_rejects_unknown() -> None:
    assert set(_PET_MOODS) == set(MOODS)
    directives = SimpleNamespace(
        mood="暴怒",
        text="",
        expression="",
        motion="",
        silent=False,
        approve=False,
        refuse=True,
    )
    cleaned = validate_directives(directives)
    assert cleaned.mood == ""
    assert cleaned.refuse is True
    assert mood_description("孤独") == "有点孤单，想要陪伴"


def test_runtime_configuration_validation_rejects_bad_mood_blocks() -> None:
    from config.loader import default_configuration_values

    defaults = default_configuration_values()
    bad = dict(defaults)
    bad["mood"] = {"decay_enabled": "sometimes", "decay_after_seconds": 14400}
    from app.runtime import validate_runtime_configuration

    configured = LoadedConfiguration(path=Path("unused.yaml"), values=dict(bad), source_digest="")
    with pytest.raises(ConfigurationError):
        validate_runtime_configuration(configured)
    bad2 = dict(defaults)
    bad2["mood"] = {"decay_enabled": True, "decay_after_seconds": 60}
    with pytest.raises(ConfigurationError):
        validate_runtime_configuration(LoadedConfiguration(Path("u2.yaml"), dict(bad2), ""))
    bad3 = dict(defaults)
    bad3["mood"] = {
        "decay_enabled": True,
        "decay_prompt_probability": 2.5,
    }
    with pytest.raises(ConfigurationError):
        validate_runtime_configuration(LoadedConfiguration(Path("u3.yaml"), dict(bad3), ""))
    bad4 = dict(defaults)
    bad4["mood"] = {
        "decay_enabled": True,
        "proactive_mood_gate": {"enabled": True, "multipliers": {"生气": 5.0}},
    }
    with pytest.raises(ConfigurationError):
        validate_runtime_configuration(LoadedConfiguration(Path("u4.yaml"), dict(bad4), ""))


def test_authoritative_vocabulary_and_aliases_shapes() -> None:
    assert MOOD_ALIASES == {"开心": "高兴", "忧郁": "难过"}
    assert sorted(MOODS) == sorted(AUTHORITATIVE_MOODS)
