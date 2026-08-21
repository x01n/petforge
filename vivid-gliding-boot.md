# 桌宠框架重建计划

## Context

当前工作树的旧项目文件处于用户已有的删除状态：原 `pyproject.toml`、`meapet/`、`docs/`、测试、启动脚本和打包文件均未保留在工作树中。用户明确要求不以旧项目作为新设计，而是参考 Yunzai 插件的 `src/` 适配器与模块化模式，从新的 `src/` 架构开始，并先在 `docs/` 中完成设计、缺口和阶段验收说明，再逐步实现。

目标是建立一个可运行、可扩展的桌宠框架，包含：

- 多渠道、多模型、多 API Key 的适配器调用与模型路由；
- 统一流式对话、思考/碎碎念/工具状态事件与输出渲染；
- 工具注册、分组、身份、参数校验、风险分级、白名单和审批绕过；
- 好感度、情绪、长期记忆、会话时间线和摘要；
- 当前窗口/进程观察、屏幕行动、桌宠移动、表情和动作控制；
- 定时任务与事件触发器；
- PySide6/Qt 6、`QOpenGLWindow`、PyOpenGL 和 `live2d-py` 的渲染边界；
- 流式 TTS，首个具体引擎为 GPT-SoVITS；
- Linux 首先可运行，随后 Windows、macOS 通过同一平台能力接口补齐近似功能。

资源事实必须按实际文件处理：`/home/clfchen/Desktop/file/meapet/temp/resources/live2d/model/mea_live2d` 没有 `.model3.json` 或 `.model.json` 模型；资源包含 GPT-SoVITS 权重和六组有效参考 WAV/TXT；六个互动 WAV 只有 12 字节 `RIFF0000WAVE` 占位数据，不能作为可播放音频；450 个 WebP 精灵可作为 Live2D 不可用时的回退资源。GPT-SoVITS 的服务协议不在资源目录中，因此框架先定义本地子进程适配器协议和缺失引擎时的文本降级，不伪造服务地址或调用参数。

## 推荐目标目录

使用物理根目录 `src/`，但应用包固定为 `src/meapet/`，避免出现裸 `src/config` 等顶级包并保持 `meapet` 的包身份；配置向导保持为 `src/wizard/`。推荐目录如下：

```text
src/
├── meapet/
│   ├── config/
│   ├── core/
│   │   ├── contracts/
│   │   ├── events/
│   │   ├── conversation/
│   │   └── tts/
│   ├── adapters/
│   │   ├── direct/
│   │   │   ├── openai_chat/
│   │   │   ├── ollama/
│   │   │   ├── responses/
│   │   │   └── anthropic/
│   │   └── agent/
│   │       ├── hermes/
│   │       ├── openclaw/
│   │       └── agent_link/
│   ├── services/
│   │   ├── conversation/
│   │   ├── model_routing/
│   │   ├── memory/
│   │   ├── affection/
│   │   ├── tools/
│   │   ├── vision/
│   │   └── scheduler/
│   ├── db/
│   ├── gui/
│   │   ├── qt6/
│   │   ├── renderers/
│   │   └── platforms/
│   ├── utils/
│   ├── logger/
│   └── __main__.py
└── wizard/
```

根目录保留极薄的兼容入口：`pet.py`、`setup_wizard.py`；入口只负责启动，不承载业务编排。

## 核心协议与模块职责

### 1. 统一消息、流事件和会话

以当前旧实现中已经存在并经过测试的协议作为迁移基线：

- `direct/types.py` 中的 `CanonicalChatRequest`、`TextDelta`、`ReasoningDelta`、`ToolCallInvocation`、`ToolCallChunk`、`UsageEvent`、`StreamDone`；
- `agent/base.py` 中的 `AgentTurnRequest`、`ImageAttachment`、`ToolStatus`、`TurnCompleted`、`TurnFailed`、`TurnCancelled`；
- `conversation/output_protocol.py` 中的 `<MEAPET_SEGMENT>` 流式解析和 `Segment` 契约；
- `conversation/timeline.py` 中按 `mode/profile_id/session_id` 隔离的时间线键；
- `conversation/orchestrator.py` 中基于 generation、turn 和会话键丢弃迟到异步事件的规则。

新 `core/contracts` 只放稳定数据协议，供应商响应和 Qt 类型不得渗入其中。统一事件流至少区分文本、reasoning、可展示碎碎念、工具调用、工具状态、审批请求、音频分片、错误、使用量和结束事件。界面按配置独立显示或隐藏 reasoning、碎碎念和工具状态，不把模型的原始隐藏推理自动当作用户可见文本。

### 2. 模型适配器与多模型路由

采用 Yunzai 的三层边界：

```text
ConversationService / ModelRouter
    -> ProviderAdapterRuntime
        -> provider request/stream converter
```

公共运行时负责历史、取消、超时、重试策略、工具循环、事件归一化和会话锁；每个提供方适配器只负责请求结构、响应结构和流事件转换。渠道配置包含渠道标识、适配器类型、基础 URL、API Key 引用、模型列表、默认模型、能力声明、超时和重试；密钥只能来自 YAML 的安全引用或环境变量，日志与 UI 脱敏。

模型路由配置驱动“按任务分配模型”，例如对话、工具规划、视觉观察、摘要、记忆提取、碎碎念和 TTS 文本预处理分别绑定模型/渠道；模型路由失败不得隐式把 Agent 协议当作 HTTP 直连回退。渠道健康、Key 轮换和备用模型属于路由服务，不暴露给桌面层。

### 3. 工具注册、执行和权限

新 `services/tools` 提供：

- 工具注册表：身份使用来源命名空间加工具名，重复身份直接拒绝；
- 工具分组和按意图最小化暴露；
- 模型可见 schema 与真实执行器分离；
- 暴露集合校验、身份解析、参数 schema 校验、调用去重和调用上限；
- 低风险自动执行，高风险确认；
- YAML 白名单、会话临时放行、绕过审批开关和过期清理；
- 默认 fail-closed：路由/审批/参数校验失败时不执行敏感工具；
- 只读观察工具与副作用工具的并发策略分离。

首批能力按实际需求拆分为：当前前台窗口和进程观察、屏幕截图/观察、桌宠位置移动、表情/动作、说话/播放、通知、定时任务、剪贴板/文件和受限命令。执行系统命令、写文件、改变窗口或控制桌面的能力默认高风险；工具参数必须再次校验路径、命令和目标，不能因模型已看到 schema 而跳过执行前检查。

继续保留已有 `CapabilityRegistry`、`CompanionControlBroker` 和控制传输层的安全边界：来源 IP、Host、Origin、Bearer Token、请求体积、速率限制、TLS/CA/mTLS；令牌长度要求为至少 32 个字符。Qt 主线程只接收标准工具/状态事件，实际工作在后台任务中完成。

### 4. 好感度、记忆和情绪

将旧 `MeaMemory` 的有效业务约束迁入服务与数据库边界，而不是把 SQLite 细节暴露给模型或界面：

- 保持现有 schema v5、聊天记录、事件、长期记忆、会话轮次、摘要、生命周期维护和导入导出能力；
- `AffectionService` 负责好感度增减、每日上限、等级和情绪变化；
- `MemoryService` 负责记忆创建、标签、重要性、相似检索、合并、衰减和摘要；
- `ConversationRepository` 负责时间线持久化；
- 记忆注入 system prompt 时使用独立长度预算，并标记来源，避免与当前对话和工具说明互相挤占；
- 用户可配置记忆开关、保留范围、导出与清理。

### 5. GPT-SoVITS 流式 TTS

定义 `SpeechRequest`、`SpeechChunk`、`SpeechResult`、`EngineHealth` 等公共协议，TTS 引擎只实现协议，不直接依赖 Qt 控件。`GsvAdapter` 读取配置指定的权重目录和参考目录，沿用已确认的参考路径格式：

```text
<ref_dir>/<mood>/<lang>_<mood>.wav
<ref_dir>/<mood>/<lang>_<mood>.txt
```

实际推理通过可取消的本地 Python 子进程适配器；由于资源仅提供模型/参考文件而没有 GPT-SoVITS 服务协议，第一阶段只约定 stdin/stdout 或明确的本地命令适配边界，命令名、参数和音频帧格式通过配置/适配器实现确定，不写死未经验证的外部服务。输出通过有界音频队列进入播放端，模型文本到达即分段送入 TTS，实现“回答尚未结束就开始说话”；引擎不可用时立即回退为文本显示，不阻塞对话。

沿用 `language_policy` 和 `TranslationService` 的职责边界。资源中的互动 WAV 在有效音频校验前不得进入播放候选。

### 6. Qt 6、OpenGL、Live2D 与平台能力

`gui/qt6` 只负责 Qt 生命周期、窗口、信号和主线程；`gui/renderers` 定义 `Renderer` 能力接口和状态；`gui/platforms` 定义统一的窗口观察、进程列表、屏幕捕获、输入/点击穿透、音频和进程启动能力。

OpenGL 渲染器使用 `QOpenGLWindow` 的 `initializeGL`、`resizeGL`、`paintGL` 生命周期；PyOpenGL 只在 Qt 当前 OpenGL context 中发出绘制命令，资源创建和销毁绑定 context。后台线程不得直接访问 QWidget/QWindow。

Live2D 适配器只把模型目录、能力探测、模型加载、参数/动作/表情调用作为能力接口；不能把旧实现中 `Idle`、`Angry`、`ParamAngleX` 等硬编码名称当作通用契约。模型目录为空或模型加载失败时，自动使用 `SpriteRenderer` 加载 `temp/resources/sprites` 的 WebP 帧；模型存在后先做模型清单和参数/动作能力探测，再建立动作映射。点击穿透、拖动和位置移动由平台窗口层控制，与渲染器解耦。

平台顺序：Linux 首先实现 X11/Wayland 环境下可验证的基础窗口、进程、前台窗口、截图和点击穿透能力；Windows 再实现等价能力；macOS 最后实现等价能力。平台接口不在核心协议中暴露平台专有类型，并为不支持的能力返回明确的能力状态，而不是静默伪成功。

### 7. 对话展示、随机行为和调度

`ConversationPresentationService` 消费统一事件：文本分片更新气泡、reasoning/碎碎念按设置进入单独区域、工具调用显示状态、审批事件显示确认卡、音频分片进入播放队列，结束时恢复待机动作。

`BehaviorService` 将模型主动行为与随机行为分离：模型只能通过权限管线调用移动、观察、表情、动作和通知工具；随机行为使用可取消、可限频的策略调度器，不能绕过工具权限，也不能与用户对话/高风险任务竞争同一执行队列。`SchedulerService` 管理定时任务；`TriggerService` 管理窗口变化、空闲时间、启动、用户交互等事件触发；所有任务必须有来源、优先级、取消和执行记录。

## YAML 配置分域

以 `config.yaml` 为主，保留一次性的配置规范化与兼容门面，分域如下：

```yaml
app: {}
llm:
  channels: []
  models: {}
  routing: {}
tools:
  groups: {}
  permissions: {}
tts: {}
rendering: {}
watcher: {}
scheduler: {}
storage: {}
ui:
  stream: {}
```

配置加载使用 PyYAML 的 `safe_load`/`safe_dump`；路径更新拒绝 `__proto__`、`constructor`、`prototype` 等危险路径，并对资源路径做绝对路径保留、相对路径锚定和存在性/类型能力检查。密钥支持环境变量占位符，配置导出默认脱敏。旧 JSON 只作为兼容读取/迁移输入，不再作为新配置主格式。

## 文档先行阶段

第一阶段先新增 `docs/` 文档，不实现未经确认的外部协议：

1. `docs/architecture.md`：目标目录、依赖方向、线程模型、事件流和模块边界；
2. `docs/adr/0001-src-layout.md`：采用 `src/meapet` 和 `src/wizard` 的原因；
3. `docs/adr/0002-adapter-runtime.md`：公共适配器运行时、提供方适配器、工具循环边界；
4. `docs/adr/0003-tool-security.md`：identity、暴露集双验、风险、审批、白名单和 fail-closed；
5. `docs/adr/0004-renderer-platform.md`：Qt6/OpenGL/Live2D/精灵回退和平台顺序；
6. `docs/adr/0005-streaming-tts.md`：GPT-SoVITS 适配器、分段、队列、降级和资源约束；
7. `docs/configuration.md`：YAML schema、渠道/模型/路由、权限和资源路径；
8. `docs/resources.md`：`temp/resources` 实际清单、缺失 Live2D 模型、无效互动 WAV 和部署要求；
9. `docs/roadmap.md`：阶段、依赖、验收、Linux/Windows/macOS 顺序；
10. `docs/testing.md`：无 GUI 合约测试、Qt 线程测试、工具安全测试、资源能力测试和运行验收。

## 分阶段实施

### 阶段 0：文档和契约冻结

完成上述文档、架构 ADR、事件类型草案、YAML schema 草案、资源能力探测说明和依赖方向检查。验收：文档中的每个公共标识符均能对应实现计划，不引入未确认的 GPT-SoVITS 或 Live2D 接口。

### 阶段 1：`src` 脚手架、uv、配置和日志

创建 `src/meapet`、`src/wizard`、`tests` 的新布局；更新 `pyproject.toml`、uv lock、pytest/ruff 配置和 Linux 首发依赖；实现 YAML 加载/规范化、资源探测、日志、入口和最小启动。禁止在本阶段恢复旧大类 `MeaPet` 的 Qt mixin 编排。

### 阶段 2：核心契约、直连/Agent 适配器门面

迁移并测试统一消息、流事件、会话时间线、取消、generation/turn 迟到事件丢弃；实现多渠道适配器注册表和模型路由；保留现有 Direct/Agent 协议兼容门面，供应商适配器只处理协议转换。

### 阶段 3：工具运行时和安全控制

实现工具 registry、identity、分组、暴露集双验、参数校验、调用去重/上限、风险分类、审批、白名单和会话绕过 TTL；接入桌宠动作/表情/位置、前台窗口/进程观察、截图和受限系统命令工具；工具状态以统一事件发送给 UI。

### 阶段 4：数据库、记忆和好感度

迁移 schema v5 约束和数据访问；拆分好感度、记忆检索、摘要、事件、会话存储服务；增加迁移、导出、清理和并发测试。

### 阶段 5：GPT-SoVITS 和流式输出

实现 TTS 公共协议、GSV 本地适配器、参考音情绪/语言选择、取消、有界音频队列、播放端和文本降级；将统一文本事件与 TTS 分段管线接入展示服务。模型/服务不可用时对话仍应完整结束。

### 阶段 6：Qt6/OpenGL/Live2D/精灵渲染

实现 QOpenGLWindow 宿主、PyOpenGL context 生命周期、透明置顶窗口、拖动/点击穿透、渲染状态、精灵回退和模型能力探测；真实 Live2D 模型出现后再补动作/表情映射，当前资源状态以精灵回退作为可运行验收。

### 阶段 7：行为、窗口观察、定时任务和触发器

实现随机行为策略、模型主动行为、前台窗口变化、空闲/交互/启动触发、定时任务、优先级队列、取消和审计记录。随机行为不得绕过权限或阻塞用户会话。

### 阶段 8：配置向导、跨平台和打包

将配置向导迁移到 `src/wizard`；Linux 完整验证后补 Windows、macOS 平台实现；更新 PyInstaller 数据/隐藏导入声明、资源能力检查和发行包文档。兼容入口只做转发。

## 关键迁移来源

仅从文件证据迁移边界和行为，不复制旧桌面总编排：

- `/home/clfchen/Desktop/file/meapet/temp/meapet/meapet/direct/types.py`、`direct/client.py`：统一直连请求/流事件；
- `/home/clfchen/Desktop/file/meapet/temp/meapet/meapet/agent/base.py`、`agent/factory.py`：Agent 稳定事件契约；
- `/home/clfchen/Desktop/file/meapet/temp/meapet/meapet/conversation/output_protocol.py`、`timeline.py`、`orchestrator.py`：输出协议、时间线和迟到事件规则；
- `/home/clfchen/Desktop/file/meapet/temp/meapet/meapet/control/capabilities.py`、`broker.py`、`transport.py`：能力注册和控制安全边界；
- `/home/clfchen/Desktop/file/meapet/temp/meapet/meapet/memory/db.py`：schema v5、好感度和记忆行为；
- `/home/clfchen/Desktop/file/meapet/temp/meapet/meapet/tts/engines/gsv.py`、`tts/language_policy.py`：GSV 参考音选择和语言策略；
- `/home/clfchen/Desktop/file/meapet/temp/meapet/meapet/desktop/renderer.py`、`live2d_widget.py`：精灵文件命名和 Live2D 加载边界；
- `/home/clfchen/Desktop/file/Yunzai/plugins/chatgpt-plugin/src/core/adapters/AbstractClient.js`、`tooling.js`、`utils/converter.js`：公共适配器运行时、工具 identity 和转换器注册表；
- `/home/clfchen/Desktop/file/Yunzai/plugins/chatgpt-plugin/src/services/tools/ToolFilterService.js`、`ToolApprovalService.js`、`ToolGroupManager.js`：工具过滤、审批、工具组调度；
- `/home/clfchen/Desktop/file/Yunzai/plugins/chatgpt-plugin/src/services/llm/LlmService.js`、`ChannelManager.js`：渠道装配、Key/模型路由和健康处理。

## 验证方案

每阶段只在对应模块完成后验证，避免把未安装的 GUI、模型或外部服务当作核心失败：

- `uv lock`、`uv sync`、`uv run pytest -q`；
- `uv run ruff check src tests`；
- `uv run python -m compileall -q src`；
- 合约测试：消息/流事件、工具调用、路由、配置 schema、时间线和迟到事件；
- 安全测试：未知 identity、未暴露工具、重复调用、危险参数、审批拒绝、白名单 TTL、fail-closed；
- 数据测试：schema v5 迁移、好感度上下限/每日上限、记忆合并/衰减/导入导出；
- TTS 测试：分段及时送入队列、取消、有界队列、GSV 缺失时文本降级、无效 WAV 不播放；
- Qt 测试：QOpenGLWindow 生命周期、无模型时精灵启动、主线程 QWidget 访问、点击穿透和窗口位置；
- Linux 端到端：启动、对话、文本流、工具状态、审批、首个音频分片播放、模型控制表情/移动、随机行为、定时触发；
- Windows/macOS：在各自平台实现完成后复用同一合约测试和能力矩阵；
- 发行验证：资源存在性/类型检查、Live2D 能力探测、PyInstaller 数据目录和隐藏导入检查。

本计划只描述推荐实现路径；在用户批准前不修改实际源码、配置或资源文件。
