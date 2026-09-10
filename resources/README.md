# MeaPet 资源目录说明

本目录只保留当前 MeaPet 运行时实际使用的资源，与源码目录分离。

## 目录总览

| 目录 | 体积（约） | 用途 |
|---|---|---|
| `models/` | 277M | GPT-SoVITS 模型权重（当前启用引擎） |
| `GPT-Sovits/` | 2.6M | GPT-SoVITS 参考音频与文本 |
| `live2d/` | 6.5M | Live2D 前端静态资源与模型目录 |
| `sprites/` | 36M | WebP 精灵桌宠回退资源（450 个文件） |

## 各目录详细说明

### `models/` — GPT-SoVITS 权重

当前 `tts.backend: gpt_sovits_stdio` 本地 worker 可使用的推理权重：

```text
models/
├── GPT_weights/mea_pro-e50.ckpt          # GPT 权重（tts.gpt_model）
└── SoVITS_weights/mea_pro_e24_s13704.pth # SoVITS 权重（tts.sovits_model）
```

对应配置键：`tts.gpt_path`、`tts.sovits_path`。由 `services.tts.worker` 或外部 GPT-SoVITS 服务加载。

### `GPT-Sovits/` — GPT-SoVITS 参考音频

按情绪分目录存放零样本/少样本推理参考音频及对应转写文本：

```text
GPT-Sovits/
├── normal/  # 常规语气
├── soft/    # 轻柔语气
└── clam/    # 平静语气（每个目录含 wav + txt，共 12 组）
```

对应配置键：`tts.ref_dir`、`tts.profiles`。参考音频按语言和角色路由。

### `live2d/` — Live2D 静态资源

```text
live2d/
├── index.html       # 当前橙色猫猫 Web 调试入口
├── js/              # Pixi、Cubism Core、pixi-live2d-display 压缩库
└── model/mea_live2d/  # 当前橙色猫猫 model3、MOC3、纹理和 A/B motion3
```

当前运行时入口为 `model/mea_live2d/橙色猫猫.model3.json`；Mare 只在 `temp/resources/` 中作为历史参考。
桌面原生渲染使用 `live2d-py + QOpenGLWindow`，Linux 默认由 Qt WebEngine/Pixi bridge 承载；模型文件缺失时启动会回退到 `sprites/`。

### `sprites/` — 精灵桌宠回退资源

450 个 `.webp` 帧图，Live2D 不可用或设置 `MEAPET_FORCE_PNG=1` 时由 `meapet/desktop/renderer.py` 渲染。

对应配置键：`sprite_dir`、`display.scale`、`display.size_factor`、`character`。

## 注意

旧 VITS/OpenJTalk 资源、无效互动 WAV、运行时字体缓存和未接入的 Live2D 调试文件已清理；
它们不属于当前代码路径。TTS 缓存写入 `data/`，不会再放入本资源目录。
