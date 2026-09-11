"""读取独立构建的桌宠控制台；运行时不依赖前端工具链。"""

from importlib.resources import files


def console_html() -> str:
    """返回包含本地样式和脚本的 shadcn/ui 控制台页面。"""

    document = files("gui.web").joinpath("static", "console", "index.html")
    if not document.is_file():
        raise RuntimeError("console frontend is not built; run npm run build in frontend/console")
    return document.read_text(encoding="utf-8")
