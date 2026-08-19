# live2d 目录占位说明

`live2d/model/mea_live2d/` 是 Live2D 模型目录（当前为空，模型文件需单独下载）。

运行时引擎：`live2d/index.html` + `live2d/js/*` 是网络渲染前端（被 PyQtWebEngine
加载），模型文件可通过以下任一方式放入：

- 直接在 `live2d/model/mea_live2d/` 放置 Live2D Cubism `.model3.json` 模型资源；
- 或通过 GUI 配置中心设置 `live2d.model_dir` 指向任意模型目录。

注意：`live2d/` 已加入 `.gitignore`（体积大、可重新获取），需要随源码分发时
请改用 Git LFS 或外部资产托管。