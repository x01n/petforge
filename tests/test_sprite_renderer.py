from __future__ import annotations

import shutil
from pathlib import Path

from gui.renderers.sprite import SpriteRenderer

RESOURCE_SPRITES = Path(__file__).resolve().parents[1] / "resources" / "sprites"


def test_sprite_defaults_to_the_stable_default_difference() -> None:
    renderer = SpriteRenderer(RESOURCE_SPRITES)

    assert renderer.outfit == "01"
    assert renderer.direction == "A"
    assert renderer.expression_code == "001"
    assert renderer.current_frame is not None
    assert renderer.current_frame.name == "mea01A_001.webp"

    renderer.advance(30.0)
    assert renderer.current_frame.name == "mea01A_001.webp"
    assert renderer.state.frame_index == 0


def test_expression_selection_is_fixed_and_does_not_cycle_other_frames() -> None:
    renderer = SpriteRenderer(RESOURCE_SPRITES)

    assert renderer.set_expression("happy")
    assert renderer.expression_code == "182"
    assert renderer.current_frame.name == "mea01A_182.webp"
    renderer.advance(30.0)
    assert renderer.current_frame.name == "mea01A_182.webp"

    assert renderer.set_expression("301")
    assert renderer.state.expression == "301"
    assert renderer.current_frame.name == "mea01A_301.webp"


def test_semantic_expressions_never_alias_blink_frames() -> None:
    """表情按钮不能误选闭眼眨眼差分。"""

    renderer = SpriteRenderer(RESOURCE_SPRITES)
    for name in ("happy", "talking", "sad", "curious", "surprised", "shy"):
        assert renderer.set_expression(name)
        assert renderer.expression_code not in {"011", "012"}

    assert renderer.set_expression("happy")
    assert renderer.current_frame.name == "mea01A_182.webp"
    assert renderer.set_expression("shy")
    assert renderer.current_frame.name == "mea01A_171.webp"
    assert renderer.set_expression("sad")
    assert renderer.current_frame.name == "mea01A_701.webp"


def test_blink_is_an_explicit_one_shot_motion() -> None:
    renderer = SpriteRenderer(RESOURCE_SPRITES)
    assert renderer.current_frame.name == "mea01A_001.webp"
    assert renderer.play_motion("blink")
    assert renderer.current_frame.name == "mea01A_011.webp"

    renderer.advance(renderer.frame_interval)
    assert renderer.current_frame.name == "mea01A_012.webp"
    renderer.advance(renderer.frame_interval)
    assert renderer.current_frame.name == "mea01A_001.webp"
    renderer.advance(30.0)
    assert renderer.current_frame.name == "mea01A_001.webp"


def test_missing_motion_difference_keeps_the_current_expression() -> None:
    renderer = SpriteRenderer(RESOURCE_SPRITES)
    assert renderer.set_expression("happy")
    assert renderer.play_motion("wave")
    assert renderer.current_frame.name == "mea01A_182.webp"
    # 资源中没有 wave 动作帧时，使用短暂的零拉伸平移补偿，仍能给用户
    # 可见反馈；原表情帧和透明画布尺寸保持不变。
    assert renderer.visual_transform == (0.0, 0.0)
    renderer.advance(0.2)
    assert renderer.visual_transform[1] < 0.0
    renderer.advance(30.0)
    assert renderer.current_frame.name == "mea01A_182.webp"
    assert renderer.state.motion == "idle"


def test_direction_switch_uses_the_same_expression_and_clears_old_motion() -> None:
    renderer = SpriteRenderer(RESOURCE_SPRITES)
    assert renderer.play_motion("blink")
    renderer.direction = "B"

    assert renderer.direction == "B"
    assert renderer.current_frame.name == "mea01B_001.webp"
    assert renderer.state.motion == "idle"
    assert renderer.visual_transform == (0.0, 0.0)


def test_invalid_direction_does_not_leave_a_missing_frame() -> None:
    renderer = SpriteRenderer(RESOURCE_SPRITES)
    renderer.direction = "Z"

    assert renderer.direction == "A"
    assert renderer.current_frame.name == "mea01A_001.webp"


def test_unstructured_webp_directory_still_has_a_safe_fallback(tmp_path: Path) -> None:
    source = RESOURCE_SPRITES / "mea01A_001.webp"
    frame = tmp_path / "frame.webp"
    frame.write_bytes(source.read_bytes())
    renderer = SpriteRenderer(tmp_path)

    assert renderer.current_frame == frame
    # 无结构化差分编号时只能绘制安全回退帧，不能把未实际渲染的表情
    # 回报为成功。
    assert not renderer.set_expression("happy")
    assert renderer.current_frame == frame
    renderer.advance(30.0)
    assert renderer.current_frame == frame


def test_sprite_capabilities_only_list_drawable_expression_and_blink_frames(tmp_path: Path) -> None:
    source = RESOURCE_SPRITES / "mea01A_001.webp"
    frame = tmp_path / "mea01A_001.webp"
    frame.write_bytes(source.read_bytes())
    renderer = SpriteRenderer(tmp_path)

    assert renderer.capabilities.expressions == ("neutral",)
    assert renderer.capabilities.motions == ("idle", "wave", "walk")
    assert renderer.supports_expression("happy") is False
    assert renderer.supports_motion("blink") is False
    assert renderer.supports_motion("wave") is True


def test_sprite_resource_reload_is_atomic_and_rescans_same_directory(tmp_path: Path) -> None:
    source = RESOURCE_SPRITES / "mea01A_001.webp"
    first_root = tmp_path / "first"
    second_bundle = tmp_path / "bundle"
    second_root = second_bundle / "sprites"
    invalid_root = tmp_path / "invalid"
    first_root.mkdir()
    second_root.mkdir(parents=True)
    invalid_root.mkdir()
    shutil.copyfile(source, first_root / "first.webp")
    shutil.copyfile(source, second_root / "second.webp")
    (invalid_root / "broken.webp").write_bytes(b"not-webp")
    renderer = SpriteRenderer(first_root)
    original_frame = renderer.current_frame

    rejected = renderer.reload_resources(invalid_root, sprite_scale=0.7)

    assert rejected["status"] == "unavailable"
    assert renderer.sprite_dir == first_root.resolve()
    assert renderer.current_frame == original_frame

    changed = renderer.reload_resources(second_bundle, sprite_scale=0.7)
    assert changed["status"] == "reloaded"
    assert changed["frame_count"] == 1
    assert changed["sprite_scale"] == 0.7
    assert changed["resource_root"] == second_bundle.resolve()
    assert changed["sprite_dir"] == second_root.resolve()
    assert renderer.sprite_dir == second_root.resolve()
    assert renderer.current_frame == (second_root / "second.webp").resolve()

    # 路径不变也必须重新扫描，宿主据此清理解码缓存。
    shutil.copyfile(source, second_root / "added.webp")
    same_root = renderer.reload_resources(second_bundle)
    assert same_root["status"] == "reloaded"
    assert same_root["frame_count"] == 2


def test_sprite_resource_reload_is_busy_while_dragging(tmp_path: Path) -> None:
    renderer = SpriteRenderer(RESOURCE_SPRITES)
    renderer.set_dragging(True)

    result = renderer.reload_resources(tmp_path)

    assert result["status"] == "busy"
    assert renderer.sprite_dir == RESOURCE_SPRITES.resolve()
