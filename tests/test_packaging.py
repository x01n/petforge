from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def _pyproject() -> dict[str, object]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_entry_points_and_wheel_use_direct_src_modules() -> None:
    project = _pyproject()["project"]
    assert project["license-files"] == ["docs/licenses/QWidget-FancyUI-GPL-3.0.txt"]
    assert project["scripts"] == {
        "meapet": "app.__main__:main",
        "meapet-wizard": "wizard.__main__:main",
    }

    wheel = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]
    packages = set(wheel["packages"])
    assert "src/meapet" not in packages
    assert {
        "src/app",
        "src/config",
        "src/core",
        "src/db",
        "src/gui",
        "src/logger",
        "src/services",
        "src/wizard",
    } <= packages
    assert wheel["force-include"] == {
        "THIRD_PARTY_NOTICES.md": "gui/qt6/THIRD_PARTY_NOTICES.md",
        "docs/licenses/QWidget-FancyUI-GPL-3.0.txt": (
            "gui/qt6/licenses/QWidget-FancyUI-GPL-3.0.txt"
        ),
    }


def test_platform_optional_dependencies_are_scoped() -> None:
    project = _pyproject()["project"]
    extras = project["optional-dependencies"]
    assert extras["linux"] == [
        "psutil>=6.1.0; platform_system == 'Linux'",
        "python-xlib>=0.33; platform_system == 'Linux'",
    ]
    assert extras["windows"] == [
        "psutil>=6.1.0; platform_system == 'Windows'",
    ]


def test_sdist_has_an_explicit_local_artifact_boundary() -> None:
    sdist = _pyproject()["tool"]["hatch"]["build"]["targets"]["sdist"]
    include = set(sdist.get("include", ()))
    only_include = set(sdist.get("only-include", ()))
    exclude = set(sdist["exclude"])

    roots = include or only_include
    assert roots >= {
        "src",
        "docs",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "config",
    } or roots >= {
        "/src",
        "/docs",
        "/README.md",
        "/THIRD_PARTY_NOTICES.md",
        "/config",
    }
    assert {
        "/.claude",
        "/.serena",
        "/.git",
        "/.venv",
        "/temp",
        "/resources",
        "/data",
        "/dist",
        "/build",
    } <= exclude
    force_include = set(sdist.get("force-include", {}))
    assert {"docs", "scripts"} <= force_include
