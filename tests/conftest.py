"""pytest 全仓根 conftest：会话级环境防护与分层标记的注册说明。

第 8 轮「回归门禁反思」（M 模块，方向 1）实测复现的两项环境脆弱性：

- 模型网关密钥类变量泄漏：宿主 shell 的 MEAPET_* / ANTHROPIC_* /
  OPENAI_* / GEMINI_* 经 ``os.environ.copy()`` 进入主进程用例与子进程，
  直接点亮 ``loader._inject_environment_channel`` 注入的网关渠道，
  配置引导 / 模型发现断言随之漂移（第 6/7 轮回归曾被污染）。
- QT_QUICK_BACKEND 继承：宿主预设值会改变依赖
  ``_configure_qt_webengine_renderer`` 默认行为的 WebGL smoke 断言。

pytest 先加载 conftest、再导入任何测试模块，顶层 pop 因此先于测试代码
执行。有意不做：

- 不清理用例运行期注入的值：monkeypatch.setenv / 直接写 os.environ
  晚于本文件执行，pop 不会与之竞争；
- 不强制 QT_QPA_PLATFORM=offscreen：本仓测试自行管理 Qt 平台参数，
  部分用例需要子进程内显式 xcb，强制 offscreen 会破坏显式契约；
- 不装配全局 QApplication fixture：offscreen 单例与 xcb 子进程用例
  共存于同一 pytest 会话，全局装配会把离屏契约扩散到 xcb 用例并
  引入跨用例窗口回收副作用。

分层标记 unit/gui/xvfb/integration/e2e 已注册于 pyproject.toml 的
[tool.pytest.ini_options].markers，回填进度为零（仅注册）；脚本
scripts/meapet_quality_gate.sh 通过 pytest 参数组合做输入分组，
不改任何既有测试文件。

关于 from __future__：pop 必须位于第一个 import 之前，而 Python 不允许
普通 import 出现在注释之后、__future__ 之前（E402）。因此本文件保持
「__future__ 顶行 -> 注释与 import os -> pop」的固定形态；若未来必须在
pop 前初始化其他依赖，正确做法是把该依赖迁移到独立的
tests/conftest_deps.py 供测试导入，而不是在 pop 之前追加 import。
"""

from __future__ import annotations

import os

# 键清单来源：src/config/loader.py ``_inject_environment_channel`` 与
# ``_hydrate_explicit_channel_keys`` 实际读取的全部渠道注入键（第 9 轮
# 方向 4 按实读枚举），加 QT_QUICK_BACKEND（app.py 的 WebEngine 渲染器
# 配置依赖其缺失状态）。其中三个历史默认模型键（ANTHROPIC_SMALL_FAST_
# MODEL、ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL）与
# ANTHROPIC_DEFAULT_HAIKU_URL 全仓零读取（rg 实查），已从清单移除。
# 有意不 pop 测试开关键：MEAPET_REAL_TTS_SMOKE / MEAPET_X11_TOPMOST_
# SMOKE / MEAPET_MCP_TEST_MARKER / MEAPET_TEST_ROOT / MEAPET_ASR_MODEL_
# DIR 属于 skipif 或测试入口参数，pop 会误伤用例收集；MEAPET_CONFIG
# 由 loader 读取但在测试内显式受控。
# 语义是 pop（移除）而不是 setdefault（写入）：我的测试会话不继承宿主
# 环境；os.environ.pop(key, None) 对缺失键返回 None、不报错。
_PROTECTED_ENV_KEYS: tuple[str, ...] = (
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

for _key in _PROTECTED_ENV_KEYS:
    os.environ.pop(_key, None)
