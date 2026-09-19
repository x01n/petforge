"""第 9 轮方向 4（桌面生命周期与测试门禁）新测试。

覆盖：C 单实例锁（QLockFile 安装/重入/陈旧恢复/跳过开关）、D 冻结重启
参数、G LFS 指针判重、F resource_version 覆盖 physics/pose 引用、A 防护
污染实证（注入键全污染时 test_startup_config 仍全绿）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import gui.qt6.app as app_module
from config.loader import LoadedConfiguration, default_configuration_values
from config.resources import _file_is_nonempty
from core.rendering.resources import (
    Live2DModelFiles,
    live2d_model_resource_version,
)
from gui.qt6.app import (
    _install_single_instance_lock,
    _process_is_alive,
    _release_single_instance_lock,
    _restart_launch_arguments,
    _single_instance_lock_path,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 与 tests/conftest.py 的 _PROTECTED_ENV_KEYS 同集合；本文件顺序独立。
_POLLUTION_KEYS = (
    "MEAPET_API_BASE",
    "MEAPET_BASE_URL",
    "MEAPET_CHANNEL_ID",
    "MEAPET_PROTOCOL",
    "MEAPET_MODEL",
    "MEAPET_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_API_KEY",
    "GEMINI_BASE_URL",
    "GEMINI_MODEL",
    "GEMINI_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "QT_QUICK_BACKEND",
)


@pytest.fixture(autouse=True)
def _reset_single_instance_holder(monkeypatch: pytest.MonkeyPatch) -> object:
    """每个用例前后清空模块级锁持有者，避免跨用例泄漏锁。

    强制移除 MEAPET_SINGLE_INSTANCE，防止宿主环境把它设为 0 时单实例
    用例全部跳过锁路径而假绿。
    """

    monkeypatch.delenv("MEAPET_SINGLE_INSTANCE", raising=False)
    _release_single_instance_lock()
    yield
    _release_single_instance_lock()


def _configuration_with_database(database: Path) -> LoadedConfiguration:
    values = default_configuration_values(resource_root=_PROJECT_ROOT / "resources")
    values["storage"] = {"database": str(database)}
    return LoadedConfiguration(Path("unused.yaml"), values)


@pytest.mark.parametrize(
    ("frozen", "expected_prefix"),
    [
        (False, ["-m", "app"]),
        (True, []),
    ],
)
def test_d_restart_launch_arguments_shape(frozen: bool, expected_prefix: list[str]) -> None:
    """源码模式前缀 -m app；冻结模式直接透传子进程参数。"""

    child = ["--config", "config/app.yaml", "--restart-ready-file", "/tmp/m.json"]
    assert _restart_launch_arguments(child, frozen=frozen) == expected_prefix + child


def test_d_frozen_mode_never_adds_module_invocation() -> None:
    """冻结产物参数向量中不得出现 -m app。"""

    result = _restart_launch_arguments(
        ["--restart-ready-file", "/tmp/m.json"],
        frozen=True,
    )
    assert "-m" not in result
    assert "app" not in result


def test_c_lock_path_abs_is_database_neighbor_and_fallback_is_temp() -> None:
    """绝对 storage.database 的锁是邻居文件；空路径回退临时目录。"""

    project_root = Path("/tmp/meapet-round9-data")
    assert _single_instance_lock_path(project_root / "meapet.sqlite3") == (
        project_root / "meapet.single_instance.lock"
    )
    fallback = _single_instance_lock_path(Path(""))
    assert fallback.is_absolute()
    assert fallback != Path()


def test_c_install_uses_storage_database_dir_and_holds_pid(tmp_path: Path) -> None:
    """安装锁后锁文件存在、首行是当前 PID；重入不覆盖既有持有者。"""

    configuration = _configuration_with_database(tmp_path / "db.sqlite3")
    _install_single_instance_lock(configuration)
    holder_lock = app_module._single_instance_lock_holder.get("lock")
    assert holder_lock is not None
    lock_path = tmp_path / "meapet.single_instance.lock"
    assert lock_path.is_file()
    recorded_pid = int(lock_path.read_text(encoding="utf-8").splitlines()[0].strip())
    assert recorded_pid == os.getpid()
    assert _process_is_alive(os.getpid()) is True

    _install_single_instance_lock(configuration)
    assert app_module._single_instance_lock_holder.get("lock") is holder_lock


def test_c_release_removes_lock_file_and_can_be_reacquired(tmp_path: Path) -> None:
    """释放后锁文件消失且可再次获取。"""

    configuration = _configuration_with_database(tmp_path / "db.sqlite3")
    _install_single_instance_lock(configuration)
    lock_path = tmp_path / "meapet.single_instance.lock"
    assert lock_path.is_file()
    _release_single_instance_lock()
    assert not lock_path.exists()
    _release_single_instance_lock()
    _install_single_instance_lock(configuration)
    assert lock_path.is_file()
    _release_single_instance_lock()


def test_c_stale_lock_from_dead_pid_is_recovered(tmp_path: Path) -> None:
    """锁文件指向已死 PID：清理陈旧锁后本次启动成功取锁。"""

    database = tmp_path / "db.sqlite3"
    lock_path = tmp_path / "meapet.single_instance.lock"
    dead_pid: int = _choose_dead_pid()
    lock_path.write_text(f"{dead_pid}\n", encoding="utf-8")
    assert _process_is_alive(dead_pid) is False
    configuration = _configuration_with_database(database)
    _install_single_instance_lock(configuration)
    assert app_module._single_instance_lock_holder.get("lock") is not None
    recorded = int(lock_path.read_text(encoding="utf-8").splitlines()[0].strip())
    assert recorded == os.getpid()
    _release_single_instance_lock()


def test_c_environment_switch_disables_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MEAPET_SINGLE_INSTANCE=0 时整体跳过，不触碰锁文件。"""

    monkeypatch.setenv("MEAPET_SINGLE_INSTANCE", "0")
    configuration = _configuration_with_database(tmp_path / "db.sqlite3")
    _install_single_instance_lock(configuration)
    assert app_module._single_instance_lock_holder.get("lock") is None
    assert not (tmp_path / "meapet.single_instance.lock").exists()


def _choose_dead_pid() -> int:
    """返回一个几乎必然不存在的候选 PID，跳过极端超大型号。"""

    import random

    for _ in range(24):
        candidate = random.SystemRandom().randint(200_000, 2_000_000)
        if not _process_is_alive(candidate):
            return candidate
    pytest.skip("cannot find a provably dead pid for stale lock test")


def test_g_lfs_pointer_first_line_is_treated_as_missing(tmp_path: Path) -> None:
    """首行 version https://git-lfs... 指针文件判为非真实权重。"""

    pointer = tmp_path / "weights.ckpt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
        "size 123456789\n",
        encoding="utf-8",
    )
    assert not _file_is_nonempty(pointer)


def test_g_lfs_pointer_three_line_structure_is_treated_as_missing(tmp_path: Path) -> None:
    """自定义 version 头的三行指针结构同样按缺失处理。"""

    pointer = tmp_path / "weights.pth"
    pointer.write_text(
        "version https://example.invalid/spec/v1\noid sha256:abcd\nsize 42\n",
        encoding="utf-8",
    )
    assert not _file_is_nonempty(pointer)


def test_g_real_binary_weight_and_empty_file_keep_semantics(tmp_path: Path) -> None:
    """真实字节流不误伤；空文件保持非真实。"""

    weight = tmp_path / "real.pth"
    weight.write_bytes(b"\x80\x02" + b"0" * 4096)
    assert _file_is_nonempty(weight)
    (tmp_path / "empty.ckpt").write_bytes(b"")
    assert _file_is_nonempty(tmp_path / "empty.ckpt") is False
    assert _file_is_nonempty(tmp_path / "missing.ckpt") is False


def test_f_resource_version_covers_physics_and_pose_references(tmp_path: Path) -> None:
    """FileReferences 中 physics/pose 引用变化必须改变 resource_version。"""

    descriptor = tmp_path / "model.model3.json"
    moc = tmp_path / "model.moc3"
    texture_dir = tmp_path / "model.1024"
    texture_dir.mkdir()
    texture = texture_dir / "texture_00.png"
    moc.write_bytes(b"moc-bytes")
    texture.write_bytes(b"png-bytes")
    physics = tmp_path / "model.physics3.json"
    pose = tmp_path / "model.pose3.json"
    physics.write_bytes(b'{"Version": 1, "Meta": {}}')
    pose.write_bytes(b'{"Type": "Live2D Pose"}')
    descriptor.write_text(
        json.dumps(
            {
                "Version": 3,
                "FileReferences": {
                    "Moc": "model.moc3",
                    "Textures": ["model.1024/texture_00.png"],
                    "Physics": "model.physics3.json",
                    "Pose": "model.pose3.json",
                },
                "Groups": [
                    {
                        "Target": "Parameter",
                        "Name": "ParamAngleX",
                        "Ids": ["ParamAngleX"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    files = Live2DModelFiles(
        descriptor=descriptor,
        moc=moc,
        textures=(texture,),
    )
    baseline = live2d_model_resource_version(files)
    assert baseline
    physics.write_bytes(b'{"Version": 1, "Meta": {"EffectiveForces": []}}')
    assert live2d_model_resource_version(files) != baseline
    physics.write_bytes(b'{"Version": 1, "Meta": {}}')
    assert live2d_model_resource_version(files) == baseline
    changed = live2d_model_resource_version(files)
    pose.write_bytes(b'{"Type": "Live2D Pose", "Groups": []}')
    assert live2d_model_resource_version(files) != changed


def test_a_polluted_channel_keys_keep_startup_suite_green() -> None:
    """全污染注入键时 test_startup_config 必须依旧全绿（conftest 防护实证）。

    子进程不走 uv run：conftest 的 pop 在收集阶段生效，直接以当前解释器
    污染运行，验证清单同步后的防护覆盖这些历史缺失键。
    """

    env = os.environ.copy()
    for key in _POLLUTION_KEYS:
        env[key] = f"polluted-{key}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(_PROJECT_ROOT / "tests" / "test_startup_config.py"),
        ],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"polluted run failed\nstdout tail:\n{result.stdout[-4000:]}\n"
        f"stderr tail:\n{result.stderr[-4000:]}"
    )
    assert "failed" not in result.stdout.splitlines()[-1]
