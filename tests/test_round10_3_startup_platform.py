"""第 10 轮方向 3：启动链路与跨平台健壮性的定向回归测试。

覆盖七个横切面：cwd 仓库探测显式开关、RuntimeLoop 可配置启动超时、
数据库目录创建回落、数据目录层级一致性、崩溃恢复 marker、Windows
信号分支防御式注册、快捷键桥接停止时序。全部为快层用例，不依赖真实
Qt/Xvfb/webengine。
"""

from __future__ import annotations

import asyncio
import os
import signal
import threading
from pathlib import Path

import pytest

import config.loader as loader
from app.loop import RuntimeLoop
from config.loader import ConfigurationError, discover_configuration, parse_bool


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _environment(root: Path) -> dict[str, str]:
    return {
        "HOME": str(root / "home"),
        "XDG_CONFIG_HOME": str(root / "xdg-config"),
        "XDG_DATA_HOME": str(root / "xdg-data"),
    }


# A. cwd 仓库探测显式开关
def test_ignore_cwd_repo_flag_disables_cwd_repository_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MEAPET_IGNORE_CWD_REPO=1 时启动目录里的 pyproject.toml 不产生项目根。"""

    monkeypatch.setattr(loader, "_module_project_root", lambda: None)
    detected = loader._find_project_root(
        tmp_path,
        environment={"MEAPET_IGNORE_CWD_REPO": "1"},
    )
    assert detected == tmp_path


def test_ignore_cwd_repo_flag_disables_cwd_config_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """开关打开后不采纳 cwd 的 config.yaml，用户配置目录照常发现。

    ``_find_project_root`` 本用例内受真实调用：cwd 层同时存在 config.yaml
    与 pyproject.toml，开关打开时必须跳过仓库采纳这一环。
    """

    monkeypatch.setattr(loader, "_module_project_root", lambda: None)
    _write(tmp_path / "pyproject.toml", '[project]\nname = "decoy"\n')
    _write(tmp_path / "config.yaml", "app:\n  name: decoy-repo\n")
    user_path = _write(
        tmp_path / "xdg-config" / "meapet" / "config.yaml",
        "app:\n  name: user\n",
    )
    env = _environment(tmp_path)
    env["MEAPET_IGNORE_CWD_REPO"] = "1"
    xdg_values = {key: value for key, value in env.items() if not key.startswith("XDG_")}
    xdg_values["MEAPET_IGNORE_CWD_REPO"] = "1"
    monkeypatch.setattr(
        loader,
        "_user_configuration_paths",
        lambda *, environment, home: (  # noqa: ARG005
            user_path,
        ),
    )
    selected = discover_configuration(
        environment=xdg_values,
        cwd=tmp_path,
        project_root=None,
        home=tmp_path / "home",
    )
    assert selected.path == user_path.resolve()
    assert selected.values["app"]["name"] == "user"


def test_cwd_repository_detection_default_semantics_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """开关默认关闭时既有语义不变：仍采纳 cwd 层 config.yaml。"""

    monkeypatch.setattr(loader, "_module_project_root", lambda: None)
    _write(tmp_path / "pyproject.toml", '[project]\nname = "project"\n')
    _write(tmp_path / "config.yaml", "app:\n  name: cwd-locked\n")
    env = _environment(tmp_path)
    env.pop("MEAPET_IGNORE_CWD_REPO", None)
    selected = discover_configuration(
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert selected.path == (tmp_path / "config.yaml").resolve()
    assert selected.values["app"]["name"] == "cwd-locked"


def test_ignore_cwd_repo_flag_parses_like_existing_environment_keys() -> None:
    """开关值与既有 MEAPET_* 环境键共用同一布尔解析器。"""

    assert parse_bool("1", field_name="MEAPET_IGNORE_CWD_REPO") is True
    assert parse_bool("true", field_name="MEAPET_IGNORE_CWD_REPO") is True
    assert parse_bool("no", field_name="MEAPET_IGNORE_CWD_REPO") is False
    assert parse_bool("0", field_name="MEAPET_IGNORE_CWD_REPO") is False


# B. RuntimeLoop 可配置启动超时
def test_runtime_loop_start_timeout_ui_key_accepts_range() -> None:
    """ui.runtime_start_timeout_seconds 为数字且过界被拒。"""

    from app.runtime import _validate_runtime_values

    _validate_runtime_values(
        {"ui": {"hotkeys": {"bindings": []}, "runtime_start_timeout_seconds": 25.0}}
    )
    with pytest.raises(ConfigurationError, match="runtime_start_timeout_seconds"):
        _validate_runtime_values(
            {"ui": {"hotkeys": {"bindings": []}, "runtime_start_timeout_seconds": 1000.0}}
        )


def test_runtime_loop_short_start_timeout_cancels_startup_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """超时不直接退出而是取消启动任务并回归诊断，随后 stop 仍可收敛。"""

    sanitized = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "MEAPET_CONFIG",
            "MEAPET_API_BASE",
            "MEAPET_BASE_URL",
            "MEAPET_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        }
    }
    monkeypatch.setattr(os, "environ", sanitized)
    del tmp_path  # 该用例不接触文件系统，避免 cwd 探测噪声

    class Runtime:
        def __init__(self) -> None:
            self.database = type("DB", (), {"close": lambda self: None})()
            self.cancelled = threading.Event()
            self.closed = 0

        async def start_background(self) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

        async def close(self) -> None:
            self.closed += 1

    runtime = Runtime()
    loop = RuntimeLoop(runtime)
    with pytest.raises(Exception) as error:
        loop.start(timeout=0.05)
    assert runtime.cancelled.wait(2)
    assert runtime.closed >= 1
    assert isinstance(error.value, (TimeoutError, RuntimeError))
    assert not loop.running
    loop.stop(timeout=1.0)


# C. 数据目录创建失败回落
def test_write_default_configuration_falls_back_to_home_dot_meapet_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """主路径 mkdir 抛 PermissionError 时回落到 home/.meapet-data 且写成功。"""

    fallback = tmp_path / "home" / ".meapet-data"
    fallback.mkdir(parents=True)
    target = tmp_path / "xdg-config" / "meapet" / "config.yaml"
    denied_root = tmp_path / "xdg-config"
    real_mkdir = loader.Path.mkdir

    def selective_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        if Path(self).is_relative_to(denied_root):
            raise PermissionError("denied by test")
        real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(loader.Path, "mkdir", selective_mkdir)
    monkeypatch.setattr(loader, "_user_data_directory", lambda **kwargs: fallback)
    wrote = loader._write_default_configuration(
        target,
        {"app": {"name": "fallback"}, "storage": {"database": str(fallback / "db.sqlite3")}},
    )
    assert wrote is True
    written_path = fallback / "config.yaml"
    assert written_path.is_file()
    assert "fallback" in written_path.read_text(encoding="utf-8")
    assert not target.is_file()


def test_discover_configuration_warns_and_uses_fallback_when_primary_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """配置发现整链路在主配置目录不可写时回落到数据目录顺位并成功构建。

    主写入目录（home/.config）被注入 PermissionError；``_write_default_configuration``
    随后写入与数据目录同顺位的 home/.local/share/meapet，``discover_configuration``
    的收敛分支读回该文件而不是抛出 ConfigurationError。
    """

    monkeypatch.setattr(loader, "_module_project_root", lambda: None)
    home = tmp_path / "home"
    monkeypatch.setattr(loader.Path, "home", classmethod(lambda cls: home.resolve()))
    denied_root = home / ".config"
    real_mkdir = loader.Path.mkdir

    def selective_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        if Path(self).is_relative_to(denied_root):
            raise PermissionError("denied by test")
        real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(loader.Path, "mkdir", selective_mkdir)
    env = {"HOME": str(home)}
    selected = discover_configuration(
        environment=env,
        cwd=tmp_path,
        project_root=None,
        home=home,
        create_default=True,
    )
    fallback_path = home / ".local" / "share" / "meapet" / "config.yaml"
    assert selected.values["app"]["name"] == "MeaPet"
    assert selected.path == fallback_path.resolve()
    assert not (home / ".config" / "meapet" / "config.yaml").exists()


# D. Windows 数据与配置目录层级一致
def test_windows_data_directory_matches_configuration_layers() -> None:
    """同环境常量下配置与数据路径落在同一 MeaPet 层级下（不区分大小写层）。"""

    environment = {"APPDATA": "C:/Users/demo/AppData/Roaming"}
    home = loader.Path("C:/Users/demo")
    configuration_paths = loader._user_configuration_paths(environment=environment, home=home)
    data_path = loader._user_data_directory(environment=environment, home=home)
    config_layer = configuration_paths[0].parent
    data_layer = data_path.parent
    assert config_layer.name.casefold() == data_layer.name.casefold()
    assert config_layer.parent == data_layer.parent
    assert data_path.name == "data"


def test_macos_data_directory_matches_configuration_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS 分支同样与配置层级一致（同样落 Library/Application Support）。"""

    home = loader.Path("/tmp/macos-home")
    monkeypatch.setattr(loader.sys, "platform", "darwin")
    data_path = loader._user_data_directory(environment={}, home=home)
    library_layer = home / "Library" / "Application Support"
    config_roots = [
        path.parent for path in loader._user_configuration_paths(environment={}, home=home)
    ]
    assert (library_layer / "MeaPet").resolve() in config_roots
    assert data_path == (library_layer / "MeaPet" / "data").resolve()
    assert data_path.parent.name == "MeaPet"
    assert data_path.parent.parent.resolve() == library_layer.resolve()


# E. 崩溃恢复 dirty 标记
def test_clean_shutdown_marker_lifecycle_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """干净关闭写入标记；标记存在时启动检测返回 True 且无警告。"""

    from gui.qt6.app import _warn_missing_clean_shutdown_marker, _write_clean_shutdown_marker

    monkeypatch.setattr("gui.qt6.app._RUNTIME_LIFECYCLE_MARKER", "meapet.lifecycle.test")
    database_path = tmp_path / "data" / "meapet.sqlite3"
    _write_clean_shutdown_marker(database_path)
    marker = database_path.parent / "meapet.lifecycle.test"
    assert marker.is_file()
    assert "clean-shutdown" in marker.read_text(encoding="utf-8").splitlines()[0]
    assert _warn_missing_clean_shutdown_marker(database_path) is True


def test_missing_clean_shutdown_marker_warns_but_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """标记缺失时仅记录 warning，检测返回 False，不改变启动行为。"""

    from gui.qt6.app import _warn_missing_clean_shutdown_marker

    monkeypatch.setattr("gui.qt6.app._RUNTIME_LIFECYCLE_MARKER", "meapet.lifecycle.test")
    database_path = tmp_path / "data" / "meapet.sqlite3"
    database_path.parent.mkdir(parents=True)
    with caplog.at_level("WARNING"):
        assert _warn_missing_clean_shutdown_marker(database_path) is False
    assert any("未留下干净关闭标记" in record.message for record in caplog.records)


# F. Windows 信号分支防御式注册
def test_shutdown_signal_handlers_register_windows_constants_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """monkeypatch 出 CTRL_C/CTRL_BREAK 常量后注册不抛且仍注册 SIGINT/SIGTERM。"""

    import gui.qt6.app as qt_app

    installed: list[int] = []
    handlers: dict[int, object] = {}

    def fake_getsignal(signum):
        return f"old-{signum}"

    def fake_signal(signum, handler):
        installed.append(signum)
        handlers[signum] = handler
        return handler

    monkeypatch.setattr(qt_app.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(qt_app.signal, "signal", fake_signal)
    monkeypatch.setattr(qt_app.signal, "CTRL_C_EVENT", 291, raising=False)
    monkeypatch.setattr(qt_app.signal, "CTRL_BREAK_EVENT", 292, raising=False)

    restore = qt_app._install_shutdown_signal_handlers(object())
    try:
        assert signal.SIGINT in installed
        assert signal.SIGTERM in installed
        assert 292 in installed
        handlers[signal.SIGTERM](signal.SIGTERM, None)
    finally:
        restore()


def test_shutdown_signal_handlers_unchanged_without_windows_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux 上常量缺失时行为不变：只注册 SIGINT/SIGTERM，不抛异常。"""

    import gui.qt6.app as qt_app

    for name in ("CTRL_C_EVENT", "CTRL_BREAK_EVENT"):
        monkeypatch.delattr(qt_app.signal, name, raising=False)
    installed: list[int] = []

    def fake_signal(signum, _handler):
        installed.append(signum)
        return _handler

    monkeypatch.setattr(qt_app.signal, "getsignal", lambda signum: f"old-{signum}")
    monkeypatch.setattr(qt_app.signal, "signal", fake_signal)
    restore = qt_app._install_shutdown_signal_handlers(object())
    try:
        assert signal.SIGINT in installed
        assert signal.SIGTERM in installed
        assert 291 not in installed
        assert 292 not in installed
    finally:
        restore()


# G. 快捷键桥接停止时序锁定
def test_x11_bridge_stop_requested_no_late_callback_emission() -> None:
    """X11 bridge 停止事件置位后进入的监听循环不触发回调。"""

    import gui.qt6.hotkeys as hotkeys
    from gui.qt6.hotkeys import HotkeyConfig

    binding = HotkeyConfig.from_mapping(
        {"bindings": [{"action": "open_input", "sequence": "Ctrl+Return", "mode": "hold"}]}
    ).bindings[0]
    events: list[tuple[str, bool]] = []

    class FakeRoot:
        def ungrab_key(self, keycode: int, modifier: int) -> None:
            del keycode, modifier

    class FakeDisplay:
        def __init__(self) -> None:
            self.root = FakeRoot()

        def pending_events(self) -> bool:
            return False

        def fileno(self) -> int:
            return 0

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    display = FakeDisplay()
    bridge = hotkeys._X11GlobalHotkey(
        (binding,), lambda action, active: events.append((action, active))
    )
    bridge._display = display
    bridge._root = display.root
    bridge._registrations = [(42, 4, binding)]
    bridge._stop.set()
    bridge._run()
    assert events == []
    bridge.close()
    assert bridge._registrations == []


def test_win32_bridge_close_unregisters_and_stops_before_bridge_destroy() -> None:
    """Win32 bridge close 后桥接状态清空：不遗留线程与注册表。"""

    import gui.qt6.hotkeys as hotkeys
    from gui.qt6.hotkeys import HotkeyConfig

    binding = HotkeyConfig.from_mapping(
        {"bindings": [{"action": "open_input", "sequence": "Ctrl+Return"}]}
    ).bindings[0]
    bridge = hotkeys._Win32GlobalHotkey((binding,), lambda action, active: None)
    bridge.close()
    assert bridge._thread is None
    assert bridge._registrations == []
    assert bridge._active_ids == set()
