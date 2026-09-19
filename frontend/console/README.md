# MeaPet HTML 控制台

React、TypeScript、Tailwind CSS 与官方 shadcn/ui（Radix）组件构建的独立中文前端。
包含总览、流式对话、模型与语音、桌宠动作、记忆活动和配置状态页面，支持明暗主题。
桌面布局使用始终可见的固定侧栏，主内容独立滚动；窄窗口只收窄侧栏，不切换为抽屉或覆盖层，
避免导航状态和 WebEngine 重排开销。组件源码位于 `src/components/ui`，可直接定制。

shadcn/ui 组件按 MIT 许可使用，许可文本随构建页面放在
`../../src/gui/web/static/console/SHADCN-LICENSE.txt`。

## 开发与构建

在本目录执行：

```sh
npm ci
npm run dev
npm test
npm run build
```

构建输出为 `../../src/gui/web/static/console/index.html`，CSS/JS 内联，不依赖 CDN。
Python 安装包携带该文件；用户运行桌宠不需要 Node.js。
修改前端后运行 `npm run build` 并重新打开桌宠，载入最新页面。

## 桌面接线

托盘的“打开网页控制台”载入本页面。独立 `npm run dev` 预览不连接真实桌宠，
因此显示“未连接桌宠”，控制动作保持禁用。

- `gui.qt6.web_console.WebControlSurfaceWindow` 仅持有 WebEngine 窗口和桥接。
- Qt 内置 `qrc:///qtwebchannel/qwebchannel.js` 提供 `meapetControlSurfaceBridge`。
- `getState(callback)` 和 `invoke(kind, JSON.stringify(payload), callback)` 使用现有公开协议。
- 宿主通过 `window.meapetControlSurface.setState(state)` 推送状态，前端拒绝旧版本。
- `src/lib/contract.ts` 校验状态；`src/lib/bridge.ts` 管理调用、去重与超时。
- 配置模型、完整配置编辑、诊断仍通过按钮进入既有 Qt 页面，保留已有功能。

旧网页兼容入口的模板位于 `src/gui/web/templates/control_surface.html`；Python
只注入已过滤的 JSON 状态和主题，不再存放大段 HTML。新控制台界面在本前端目录维护。
