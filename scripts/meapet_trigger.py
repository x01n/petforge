"""从编辑器/代理脚本触发桌宠输入框或提交一条消息。

使用前由宿主通过 ``local_web_server_sink`` 提供当前地址和 capability，
再在调用环境设置 ``MEAPET_LOCAL_API_URL`` 与 ``MEAPET_LOCAL_API_TOKEN``。
脚本只使用 Python 标准库，适合 VSCode task、Claude Code hook 和 Codex
外部命令；不会猜测或读取任何编辑器内部协议。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request(kind: str, payload: dict[str, object]) -> dict[str, object]:
    base_url = str(os.environ.get("MEAPET_LOCAL_API_URL", "") or "").strip().rstrip("/")
    token = str(os.environ.get("MEAPET_LOCAL_API_TOKEN", "") or "").strip()
    if not base_url or not token:
        raise RuntimeError("MEAPET_LOCAL_API_URL and MEAPET_LOCAL_API_TOKEN are required")
    body = json.dumps({"kind": kind, "payload": payload}, ensure_ascii=False).encode("utf-8")
    try:
        request = Request(
            f"{base_url}/api/action",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-MeaPet-Capability": token,
            },
            method="POST",
        )
        with urlopen(request, timeout=3.0) as response:  # noqa: S310 - URL comes from explicit local env
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"MeaPet local action failed: {type(exc).__name__}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("MeaPet local action returned an invalid response")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="触发 MeaPet 桌宠输入或消息")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--open-input", action="store_true", help="打开桌宠输入框")
    group.add_argument("--text", help="打开对话并提交文本")
    args = parser.parse_args(argv)
    kind = "open_input" if args.open_input else "submit_text"
    payload = {} if args.open_input else {"text": str(args.text)}
    try:
        result = _request(kind, payload)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if str(result.get("status", "")).lower() in {"completed", "requested"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
