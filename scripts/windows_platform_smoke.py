"""输出 Windows 平台能力探测结果。

该脚本只探测 ctypes/Qt/可选 psutil 能力，不创建窗口、不移动鼠标、不注入点击，
也不读取屏幕内容。非 Windows 环境返回 ``skipped``，便于跨平台质量门复用。
"""

# ruff: noqa: I001

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui.platforms.factory import create_desktop_platform  # noqa: E402


def main() -> int:
    if sys.platform != "win32":
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "platform": sys.platform,
                    "reason": "Windows-only platform probe",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    try:
        platform = create_desktop_platform()
        snapshot = platform.probe()
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "platform": sys.platform,
                    "reason": f"{type(exc).__name__}",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1

    capabilities = {
        item.name: {
            "available": bool(item.available),
            "state": str(item.state),
            "evidence": list(item.evidence),
        }
        for item in snapshot.capabilities
    }
    unavailable = sorted(
        name for name, item in capabilities.items() if item["state"] == "unavailable"
    )
    status = "ok" if snapshot.backend == "windows" and not unavailable else "degraded"
    print(
        json.dumps(
            {
                "status": status,
                "platform": sys.platform,
                "backend": snapshot.backend,
                "capabilities": capabilities,
                "unavailable": unavailable,
                "details": dict(snapshot.details),
                "scope": "probe_only",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if snapshot.backend == "windows" else 1


if __name__ == "__main__":
    raise SystemExit(main())
