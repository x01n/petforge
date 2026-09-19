from __future__ import annotations

from pathlib import Path

import pytest

from config.loader import (
    ConfigurationError,
    default_configuration_values,
    load_configuration,
    parse_bool,
    redact_secrets,
)
from config.resources import inspect_resources
from gui.renderers import assets
from gui.renderers.assets import (
    RuntimeCapability,
    RuntimeCapabilityState,
    probe_live2d_runtime,
    probe_render_assets,
    select_renderer,
)
from gui.renderers.sprite import SpriteRenderer
from gui.renderers.threed import Reserved3DRenderer, VulkanRuntimeProbe


def test_yaml_environment_and_secret_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        "\n".join(
            (
                "app:",
                "  name: demo",
                "llm:",
                "  channels:",
                "    - id: main",
                "      protocol: openai_chat",
                "      api_key: ${MEAPET_TEST_KEY}",
                "",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEAPET_TEST_KEY", "secret-value")
    loaded = load_configuration(config_path)
    assert loaded.values["llm"]["channels"][0]["api_key"] == "secret-value"
    assert redact_secrets(loaded.values)["llm"]["channels"][0]["api_key"] == "***"


def test_yaml_rejects_forbidden_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("app:\n  constructor: bad\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_configuration(config_path)


def test_parse_bool_does_not_treat_quoted_false_as_true() -> None:
    assert parse_bool("false", field_name="test") is False
    assert parse_bool("true", field_name="test") is True
    with pytest.raises(ConfigurationError):
        parse_bool("maybe", field_name="test")


def test_rendering_backend_configuration_has_auto_default_and_strict_values(tmp_path: Path) -> None:
    defaults = default_configuration_values()
    assert defaults["rendering"]["backend"] == "auto"
    assert defaults["rendering"]["frame_rate"] == 60.0
    assert defaults["rendering"]["geometry_audit_hz"] == 30.0
    config_path = tmp_path / "backend.yaml"
    config_path.write_text("rendering:\n  backend: vllank\n", encoding="utf-8")
    loaded = load_configuration(config_path, defaults=defaults, environment={})
    assert loaded.values["rendering"]["backend"] == "vllank"
    config_path.write_text("rendering:\n  backend: vulkan\n", encoding="utf-8")
    loaded = load_configuration(config_path, defaults=defaults, environment={})
    assert loaded.values["rendering"]["backend"] == "vulkan"
    config_path.write_text("rendering:\n  backend: directx\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="rendering.backend"):
        load_configuration(config_path, defaults=defaults, environment={})


def test_real_resource_inventory_uses_capability_facts() -> None:
    inventory = inspect_resources(Path(__file__).parents[1] / "temp" / "resources")
    assert inventory.sprite_count == 450
    assert inventory.live2d_available
    assert any(path.name == "Mare.model3.json" for path in inventory.live2d_models)
    assert any(path.name == "G_latest.pth" for path in inventory.gsv_weight_files)
    assert len(inventory.reference_voices) == 6
    assert inventory.reference_voices and all(item.valid for item in inventory.reference_voices)
    assert len(inventory.interaction_voices) == 6
    assert all(not item.valid for item in inventory.interaction_voices)


def test_live2d_runtime_loader_failure_keeps_auto_unavailable(tmp_path: Path) -> None:
    resource_root = tmp_path / "resources"
    model_root = resource_root / "live2d" / "model" / "demo"
    sprite_root = resource_root / "sprites"
    model_root.mkdir(parents=True)
    sprite_root.mkdir(parents=True)
    (model_root / "demo.moc3").write_bytes(b"moc")
    (model_root / "texture.png").write_bytes(b"texture")
    (model_root / "demo.model3.json").write_text(
        '{"FileReferences":{"Moc":"demo.moc3","Textures":["texture.png"]}}',
        encoding="utf-8",
    )
    (sprite_root / "frame.webp").write_bytes(b"RIFF\x04\x00\x00\x00WEBP")

    def broken_import(_name: str):
        raise RuntimeError("native loader failure")

    runtime = probe_live2d_runtime(importer=broken_import)
    assert not runtime.available
    selection = select_renderer(resource_root, runtime=runtime)
    assert selection.backend == "unavailable"
    assert "Live2D" in selection.reason


def test_live2d_runtime_rejects_module_without_instantiable_model_loader() -> None:
    """模块级 v3 符号齐全时仍必须验证真实模型对象契约。"""

    fake_module = type(
        "FakeLive2DModule",
        (),
        {
            "init": staticmethod(lambda: None),
            "glInit": staticmethod(lambda: None),
            "clearBuffer": staticmethod(lambda: None),
            "dispose": staticmethod(lambda: None),
            "LAppModel": object,
        },
    )
    runtime = probe_live2d_runtime(importer=lambda _name: fake_module)
    assert runtime.available is False
    assert "LAppModel" in runtime.detail
    assert "LoadModelJson" not in runtime.evidence


def test_renderer_auto_never_uses_sprite_runtime_fallback(tmp_path: Path) -> None:
    resource_root = tmp_path / "resources"
    sprite_root = resource_root / "sprites"
    sprite_root.mkdir(parents=True)
    (sprite_root / "frame.webp").write_bytes(b"RIFF\x04\x00\x00\x00WEBP")
    unavailable_native = RuntimeCapability(
        "live2d_runtime", RuntimeCapabilityState.UNAVAILABLE, "native unavailable"
    )
    unavailable_web = RuntimeCapability(
        "web_live2d", RuntimeCapabilityState.UNAVAILABLE, "web unavailable"
    )

    automatic = select_renderer(
        resource_root,
        requested_backend="auto",
        runtime=unavailable_native,
        web_runtime=unavailable_web,
    )
    explicit_sprite = select_renderer(
        resource_root,
        requested_backend="sprite",
        runtime=unavailable_native,
        web_runtime=unavailable_web,
    )
    assert automatic.backend == "unavailable"
    assert automatic.allows_runtime_fallback is False
    assert explicit_sprite.allows_runtime_fallback is False

    model_root = resource_root / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    (model_root / "demo.moc3").write_bytes(b"moc")
    (model_root / "texture.png").write_bytes(b"texture")
    (model_root / "demo.model3.json").write_text(
        '{"FileReferences":{"Moc":"demo.moc3","Textures":["texture.png"]}}',
        encoding="utf-8",
    )
    available_web = RuntimeCapability(
        "web_live2d", RuntimeCapabilityState.AVAILABLE, "web available"
    )
    explicit_web = select_renderer(
        resource_root,
        requested_backend="web_live2d",
        runtime=unavailable_native,
        web_runtime=available_web,
    )
    assert explicit_web.backend == "web_live2d"
    assert explicit_web.allows_runtime_fallback is False


def test_renderer_auto_prefers_current_web_live2d_when_both_paths_are_available(
    tmp_path: Path,
) -> None:
    resource_root = tmp_path / "resources"
    model_root = resource_root / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    (model_root / "demo.moc3").write_bytes(b"moc")
    (model_root / "texture.png").write_bytes(b"texture")
    (model_root / "demo.model3.json").write_text(
        '{"FileReferences":{"Moc":"demo.moc3","Textures":["texture.png"]}}',
        encoding="utf-8",
    )
    native = RuntimeCapability("live2d_runtime", RuntimeCapabilityState.AVAILABLE, "native ready")
    web = RuntimeCapability("web_live2d", RuntimeCapabilityState.AVAILABLE, "web ready")

    automatic = select_renderer(
        resource_root,
        requested_backend="auto",
        runtime=native,
        web_runtime=web,
    )
    explicit_native = select_renderer(
        resource_root,
        requested_backend="opengl",
        runtime=native,
        web_runtime=web,
    )

    assert automatic.backend == "web_live2d"
    assert automatic.reason == "auto selected the stable Web Live2D backend"
    assert explicit_native.backend == "opengl"


def test_renderer_backend_selection_is_explicit_and_vulkan_aliases_are_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_root = tmp_path / "resources"
    model_root = resource_root / "live2d" / "model" / "demo"
    sprite_root = resource_root / "sprites"
    model_root.mkdir(parents=True)
    sprite_root.mkdir(parents=True)
    (model_root / "demo.moc3").write_bytes(b"moc")
    (model_root / "texture.png").write_bytes(b"texture")
    (model_root / "demo.model3.json").write_text(
        '{"FileReferences":{"Moc":"demo.moc3","Textures":["texture.png"]}}',
        encoding="utf-8",
    )
    (sprite_root / "frame.webp").write_bytes(b"RIFF\x04\x00\x00\x00WEBP")
    unavailable_native = RuntimeCapability(
        "live2d_runtime", RuntimeCapabilityState.UNAVAILABLE, "native unavailable"
    )
    available_web = RuntimeCapability(
        "web_live2d", RuntimeCapabilityState.AVAILABLE, "web available"
    )
    monkeypatch.setattr(
        assets,
        "probe_qt_vulkan_runtime",
        lambda: VulkanRuntimeProbe(True, "Vulkan scenegraph API is available"),
    )

    assert (
        select_renderer(
            resource_root,
            requested_backend="auto",
            runtime=unavailable_native,
            web_runtime=available_web,
        ).backend
        == "web_live2d"
    )
    assert (
        select_renderer(
            resource_root,
            requested_backend="sprite",
            runtime=unavailable_native,
            web_runtime=available_web,
        ).backend
        == "sprite"
    )
    opengl = select_renderer(
        resource_root,
        requested_backend="opengl",
        runtime=unavailable_native,
        web_runtime=available_web,
    )
    assert opengl.backend == "unavailable"
    vllank = select_renderer(
        resource_root,
        requested_backend="vllank",
        runtime=unavailable_native,
        web_runtime=available_web,
    )
    assert vllank.backend == "unavailable"
    assert vllank.state.compatibility_alias == "vllank"
    assert "Vulkan rendering is unavailable" in vllank.reason
    vllakn = select_renderer(
        resource_root,
        requested_backend="vllakn",
        runtime=unavailable_native,
        web_runtime=available_web,
    )
    assert vllakn.backend == "unavailable"
    assert vllakn.state.compatibility_alias == "vllakn"
    with pytest.raises(ValueError):
        select_renderer(resource_root, requested_backend="unknown")


def test_reserved_3d_renderer_is_explicitly_unavailable() -> None:
    renderer = Reserved3DRenderer()
    assert renderer.capabilities.backend == "vulkan"
    assert renderer.capabilities.available is False
    assert renderer.initialize() is False
    assert renderer.initialized is False
    renderer.resize(420, 480, 2.0)
    assert renderer.last_size == (420, 480, 2.0)
    renderer.draw()
    renderer.shutdown()
    assert renderer.last_size is None


def test_reserved_3d_renderer_sanitizes_invalid_lifecycle_dimensions() -> None:
    renderer = Reserved3DRenderer()

    renderer.resize(float("nan"), float("inf"), float("-inf"))

    assert renderer.last_size == (1, 1, 1.0)
    assert renderer.initialize(object()) is False
    assert renderer.initialized is False


def test_reserved_3d_renderer_normalizes_alias_and_rejects_other_backends() -> None:
    renderer = Reserved3DRenderer("vllank")
    assert renderer.capabilities.backend == "vulkan"
    assert renderer.compatibility_alias == "vllank"
    with pytest.raises(ValueError, match="vulkan"):
        Reserved3DRenderer("opengl")


def test_empty_sprite_frames_do_not_claim_rendering_capability(tmp_path: Path) -> None:
    resource_root = tmp_path / "resources"
    sprite_root = resource_root / "sprites"
    sprite_root.mkdir(parents=True)
    (sprite_root / "empty.webp").write_bytes(b"")

    inventory = inspect_resources(resource_root)
    assert inventory.sprite_count == 0
    assert not inventory.sprite_available
    assert not SpriteRenderer(sprite_root).capabilities.available


def test_invalid_webp_frames_do_not_claim_rendering_capability(tmp_path: Path) -> None:
    resource_root = tmp_path / "resources"
    sprite_root = resource_root / "sprites"
    sprite_root.mkdir(parents=True)
    (sprite_root / "fake.webp").write_bytes(b"not-a-webp")

    inventory = inspect_resources(resource_root)
    render_assets = probe_render_assets(resource_root)
    assert inventory.sprite_count == 0
    assert not inventory.sprite_available
    assert not render_assets.sprite_available
    assert not SpriteRenderer(sprite_root).capabilities.available


def test_live2d_descriptor_requires_references_and_confined_files(tmp_path: Path) -> None:
    resource_root = tmp_path / "resources"
    model_root = resource_root / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    outside_moc = model_root.parent / "outside.moc3"
    outside_moc.write_bytes(b"moc")
    (model_root / "missing.model3.json").write_text('{"Version": 3}', encoding="utf-8")
    (model_root / "escape.model3.json").write_text(
        '{"FileReferences":{"Moc":"../outside.moc3","Textures":["texture.png"]}}',
        encoding="utf-8",
    )
    (model_root / "legacy.model.json").write_text(
        '{"FileReferences":{"Moc":"outside.moc3","Textures":["texture.png"]}}',
        encoding="utf-8",
    )

    inventory = inspect_resources(resource_root)
    render_assets = probe_render_assets(resource_root)
    assert not inventory.live2d_available
    assert not render_assets.live2d_available
    assert any(".model3.json descriptors are invalid" in item for item in inventory.warnings)
    assert any(".model.json descriptors are unsupported" in item for item in inventory.warnings)
    assert any(".model3.json descriptor(s) are invalid" in item for item in render_assets.warnings)
