from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.loader import ConfigurationError

from .configuration import ConfigurationWizard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meapet-wizard")
    parser.add_argument("--config", default="config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("validate")
    subparsers.add_parser("show")
    channel_parser = subparsers.add_parser("channel")
    channel_subparsers = channel_parser.add_subparsers(dest="channel_command", required=True)
    add_channel = channel_subparsers.add_parser("add")
    add_channel.add_argument("--id", dest="channel_id", required=True)
    add_channel.add_argument("--protocol", default="openai_chat")
    add_channel.add_argument("--base-url", required=True)
    add_channel.add_argument("--model", required=True)
    add_channel.add_argument("--api-key-env", default="")
    add_channel.add_argument("--capabilities", default="streaming,tools")
    add_channel.add_argument("--priority", type=int, default=10)
    add_channel.add_argument("--disabled", action="store_true")
    remove_channel = channel_subparsers.add_parser("remove")
    remove_channel.add_argument("--id", dest="channel_id", required=True)
    args = parser.parse_args(argv)
    wizard = ConfigurationWizard(Path(args.config))
    try:
        if args.command == "init":
            wizard.create()
            print(wizard.path)
            return 0
        if args.command == "channel":
            if args.channel_command == "add":
                path = wizard.configure_channel(
                    channel_id=args.channel_id,
                    protocol=args.protocol,
                    base_url=args.base_url,
                    model=args.model,
                    api_key_env=args.api_key_env,
                    capabilities=args.capabilities,
                    enabled=not args.disabled,
                    priority=args.priority,
                )
            else:
                path = wizard.remove_channel(args.channel_id)
            print(path)
            return 0
        if args.command == "show":
            print(json.dumps(wizard.safe_view(), ensure_ascii=False, indent=2))
            return 0
        loaded = wizard.load()
        print(f"valid: {loaded.path}")
        return 0
    except ConfigurationError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
