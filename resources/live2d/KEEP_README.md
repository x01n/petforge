# live2d 目录说明

`live2d/model/mea_live2d/` 是 Live2D 模型目录。当前活动模型是已编译的
`橙色猫猫.model3.json`；Mare 仅保留在 `temp/resources/` 作为历史参考。

`live2d/index.html` 是可独立打开的当前模型调试页；正式 Qt WebEngine 运行时由
`gui.renderers.web_live2d` 生成页面并复用 `live2d/js/*`。当前项目代码固定扫描资源根目录下的
`live2d/model/`，因此应将完成导出的
Live2D Cubism 模型资源放入 `live2d/model/mea_live2d/`；当前代码没有读取
`live2d.model_dir` 配置项。

运行时入口使用 `橙色猫猫.model3.json`，它通过 `FileReferences.Moc` 和
`FileReferences.Textures` 校验，引用的 `.moc3` 与纹理文件均为 Cubism 编辑器实际导出
的非空文件。运行时优先加载该模型，只有运行库或 Qt WebEngine 不可用时才回退到 WebP 精灵。

注意：`live2d/` 已加入 `.gitignore`（体积大、可重新获取），需要随源码分发时请改用
Git LFS 或外部资产托管。
