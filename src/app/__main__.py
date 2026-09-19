from __future__ import annotations

import argparse
import logging
import sys

from config.loader import ConfigurationError, discover_configuration, resolve_resource_path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meapet")
    parser.add_argument(
        "--config",
        default=None,
        help="显式配置文件；未提供时按启动发现优先级自动选择",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--restart-ready-file", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # 入口模块保持轻量：验证/GUI 运行时和 Qt 相关模块只在真正需要时
    # 导入，避免 shell 补全、--help 与配置发现被大型桌面依赖拖慢。
    from logger import configure_logging

    # 先建立默认 stderr sink 以便报告配置发现错误；真正的配置应用在加载
    # YAML 后再发一次完成事件，避免启动日志重复一条相同摘要。
    configure_logging(emit_event=False)
    try:
        loaded = discover_configuration(args.config)
        try:
            configure_logging(config=loaded.values, base_directory=loaded.directory)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"logging configuration is invalid: {type(exc).__name__}"
            ) from exc
        if args.validate or args.no_gui:
            from app.runtime import validate_runtime_configuration

            validate_runtime_configuration(loaded)
        rendering = loaded.values.get("rendering", {})
        resource_value = (
            rendering.get("resource_root", "./resources")
            if isinstance(rendering, dict)
            else "./resources"
        )
        resource_root = resolve_resource_path(
            resource_value, configuration_directory=loaded.directory
        )
    except ConfigurationError as exc:
        parser.error(str(exc))
    from config.resources import inspect_resources
    from gui.renderers.assets import select_renderer

    inventory = inspect_resources(resource_root)
    rendering_backend = rendering.get("backend", "auto") if isinstance(rendering, dict) else "auto"
    rendering_model = rendering.get("model") if isinstance(rendering, dict) else None
    renderer_selection = select_renderer(
        resource_root,
        requested_backend=rendering_backend,
        requested_model=rendering_model,
    )
    for warning in inventory.warnings:
        if args.validate or args.no_gui:
            print(f"resource_warning: {warning}", file=sys.stderr)
        else:
            logger.warning("resource_warning: %s", warning)
    # 自动模式在资源不完整时允许进入诊断，让首次安装能够先看到精确
    # 的 Live2D 能力缺口；显式选择不可用后端则立即返回错误。
    if (
        not (args.validate or args.no_gui)
        and renderer_selection.requested_backend != "auto"
        and not renderer_selection.available
    ):
        parser.error(
            f"requested rendering backend '{renderer_selection.requested_backend}' is unavailable: "
            f"{renderer_selection.reason}"
        )
    if args.validate or args.no_gui:
        from services.model_routing import ModelRouter

        print(f"configuration: {loaded.path}")
        diagnostics = ModelRouter.from_mapping(loaded.values).diagnostics("dialogue")
        print(f"model_channel_ready: {bool(diagnostics.get('ready'))}")
        if not diagnostics.get("ready"):
            print(
                f"model_channel_reason: {diagnostics.get('reason', 'channel is not ready')}",
                file=sys.stderr,
            )
        print(f"live2d_available: {inventory.live2d_available}")
        print(f"sprite_count: {inventory.sprite_count}")
        print(f"rendering_backend_requested: {renderer_selection.requested_backend}")
        print(f"rendering_backend_selected: {renderer_selection.backend}")
        print(f"rendering_backend_available: {renderer_selection.available}")
        print(f"rendering_backend_reason: {renderer_selection.reason}")
        selected_model = getattr(renderer_selection.model_selection, "selected", None)
        print(f"live2d_model_selected: {getattr(selected_model, 'key', '')}")
        print(f"live2d_model_choices: {', '.join(renderer_selection.model_choices)}")
        print(f"tts_reference_count: {sum(asset.valid for asset in inventory.reference_voices)}")
        # 自动模式允许空资源目录进入诊断，方便首次安装；显式后端则必须
        # 与真实运行能力一致，否则验证命令返回成功会误导启动器继续执行。
        if renderer_selection.requested_backend != "auto" and not renderer_selection.available:
            return 2
        return 0
    try:
        from app.runtime import initialize_database

        # 在导入 Qt 前完成目录创建和幂等 schema 迁移；GUI 宿主随后会由
        # ``build_runtime`` 在同一路径再次打开连接。这样无 GUI 依赖时也能
        # 明确区分数据库权限错误与桌面依赖错误。
        initialize_database(loaded)
        from gui.qt6.app import run

        if args.restart_ready_file is None:
            return int(run(loaded, inventory) or 0)
        return int(run(loaded, inventory, restart_ready_file=args.restart_ready_file) or 0)
    except ConfigurationError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        # Qt 信号桥接通常会让事件循环自然返回；若平台仍向 Python
        # 抛出 KeyboardInterrupt，也要以标准中断码结束而不打印回溯。
        return 130
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
        parser.error(f"桌面依赖未安装或当前平台不可用: {exc}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
