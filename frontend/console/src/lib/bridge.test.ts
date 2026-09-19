import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseState } from "./contract";

describe("Qt 控制台协议", () => {
  beforeEach(() => vi.resetModules());
  afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

  async function connect() {
    const host = {
      getState: vi.fn((callback: (value: string) => void) => callback('{"revision":2}')),
      invoke: vi.fn(),
    };
    vi.stubGlobal("document", { getElementById: () => null });
    vi.stubGlobal("window", {
      qt: { webChannelTransport: {} },
      QWebChannel: class {
        constructor(_transport: unknown, callback: (value: unknown) => void) {
          callback({ objects: { meapetControlSurfaceBridge: host } });
        }
      },
    });
    const bridge = await import("./bridge");
    bridge.initializeBridge();
    return { ...bridge, host };
  }

  it("拒绝旧版本、非法 JSON 和非法版本", async () => {
    const { applyState } = await connect();
    expect(applyState({ revision: 3, interaction: { text: "新回复" } })).toBe(true);
    expect(applyState({ revision: 1, interaction: { text: "旧回复" } })).toBe(false);
    expect(applyState("invalid json")).toBe(false);
    expect(parseState({ revision: true })).toBeNull();
  });

  it("在完整校验前跳过重复版本，避免空闲轮询重复渲染", async () => {
    const { applyState } = await connect();
    expect(applyState({ revision: 3, interaction: { text: "首个状态" } })).toBe(true);
    expect(applyState({ revision: 3, interaction: { text: "不应覆盖" } })).toBe(false);
  });

  it("发送准确字段、合并重复操作，并消费完成回执", async () => {
    const { invoke, host } = await connect();
    const first = invoke("select_tts_profile", { profile: "voice-one", language: "zh" });
    expect(host.invoke).toHaveBeenCalledWith("select_tts_profile", '{"profile":"voice-one","language":"zh"}', expect.any(Function));
    expect((await invoke("select_tts_profile", { profile: "voice-one", language: "zh" })).status).toBe("pending");
    expect(host.invoke).toHaveBeenCalledTimes(1);
    host.invoke.mock.calls[0][2]('{"status":"updated","message":"声音已切换"}');
    expect((await first).message).toBe("声音已切换");
    expect(host.getState).toHaveBeenCalledTimes(2);
  });

  it("超时会释放按钮，迟到回执不会覆盖下一次操作", async () => {
    vi.useFakeTimers();
    const { invoke, host } = await connect();
    const first = invoke("show_pet");
    const late = host.invoke.mock.calls[0][2];
    await vi.advanceTimersByTimeAsync(8000);
    expect((await first).status).toBe("timeout");
    const second = invoke("show_pet");
    late('{"status":"failed"}');
    host.invoke.mock.calls[1][2]('{"status":"completed"}');
    expect((await second).status).toBe("completed");
  });

  it("失败回执后相同载荷可以重新提交", async () => {
    const { invoke, host } = await connect();
    const first = invoke("select_tts_profile", { profile: "voice-two" });
    host.invoke.mock.calls[0][2]('{"status":"unavailable","message":"通道暂时不可用"}');
    expect((await first).status).toBe("unavailable");
    // 回执完成后去重键释放：同一 payload 再次发起必须真正触达宿主。
    const second = invoke("select_tts_profile", { profile: "voice-two" });
    expect(host.invoke).toHaveBeenCalledTimes(2);
    host.invoke.mock.calls[1][2]('{"status":"updated","message":"声音已切换"}');
    expect((await second).status).toBe("updated");
  });

  it("成功回执后相同载荷同样允许重放", async () => {
    const { invoke, host } = await connect();
    const first = invoke("pet_part", { part: "head" });
    host.invoke.mock.calls[0][2]('{"status":"completed"}');
    expect((await first).status).toBe("completed");
    const replay = invoke("pet_part", { part: "head" });
    expect(host.invoke).toHaveBeenCalledTimes(2);
    host.invoke.mock.calls[1][2]('{"status":"completed"}');
    expect((await replay).status).toBe("completed");
  });

  it("localStorage 主题记忆白名单校验：非法值返回 null，用户接管后令牌不再推动明暗类", async () => {
    const storage: Record<string, string> = {};
    const classListAdd = vi.fn();
    const classListRemove = vi.fn();
    const classListToggle = vi.fn();
    vi.stubGlobal("window", {
      localStorage: {
        getItem: (key: string) => storage[key] ?? null,
        setItem: (key: string, value: string) => { storage[key] = value; },
        removeItem: (key: string) => { delete storage[key]; },
      },
    });
    vi.stubGlobal("document", {
      documentElement: {
        classList: { add: classListAdd, remove: classListRemove, toggle: classListToggle },
      },
    });
    const bridgeModule = await import("./bridge");
    // 无记忆时为 null（App 侧落到 light），非法值（"foo"、""）不得通过。
    expect(bridgeModule.loadStoredTheme()).toBeNull();
    storage["meapet-console-theme"] = "foo";
    expect(bridgeModule.loadStoredTheme()).toBeNull();
    storage["meapet-console-theme"] = "";
    expect(bridgeModule.loadStoredTheme()).toBeNull();
    storage["meapet-console-theme"] = "dark";
    expect(bridgeModule.loadStoredTheme()).toBe("dark");
    expect(bridgeModule.themeHandledByUser()).toBe(false);
    bridgeModule.applyUserTheme("light");
    expect(bridgeModule.themeHandledByUser()).toBe(true);
    expect(storage["meapet-console-theme"]).toBe("light");
    expect(classListToggle).toHaveBeenCalledWith("dark", false);
  });

  it("无桌宠连接时不能伪造成功", async () => {
    vi.stubGlobal("document", { getElementById: () => null });
    vi.stubGlobal("window", {});
    const { initializeBridge, invoke } = await import("./bridge");
    initializeBridge();
    expect((await invoke("submit_text", { text: "测试" })).status).toBe("unavailable");
  });

  it("审计聚合坏数据不破坏整体状态解析", async () => {
    const { applyState } = await connect();
    // summary 结构损坏时整体状态仍可解析，审计区回退为明细显示。
    const accepted = applyState({
      revision: 5,
      diagnostics: {
        api_audit: {
          status: "available",
          count: 1,
          summary: { total: "bad", by_status: [{ key: 1, count: "x" }], latency_ms: { avg_first: "不对" } },
          records: [{ kind: "model", status: "completed" }],
        },
        logs: { status: "idle", count: 0, records: [] },
      },
    });
    expect(accepted).toBe(true);
  });

  it("审计聚合缺失时回退为仅明细显示且状态解析成功", async () => {
    const { applyState } = await connect();
    const accepted = applyState({
      revision: 6,
      diagnostics: {
        api_audit: { status: "available", count: 1, records: [{ kind: "model", status: "completed" }] },
        logs: { status: "idle", count: 0, records: [] },
      },
    });
    expect(accepted).toBe(true);
  });

  it("渲染帧耗时与日志聚合缺失或坏数据都不破坏状态解析", async () => {
    const frameTime = parseState({
      revision: 7,
      renderer: { frame_time_ms: 16.7, large_frame_deltas: 3 },
      diagnostics: {
        api_audit: {},
        logs: { summary: "broken", records: [] },
      },
    });
    expect(frameTime?.renderer.frame_time_ms).toBe(16.7);
    expect(frameTime?.renderer.large_frame_deltas).toBe(3);
    // summary 为非法结构时整体回退为 null，不导致整页解析失败。
    expect(frameTime?.diagnostics.logs.summary).toBeNull();
  });

  it("帧耗时缺失时回退空值且日志摘要坏结构不破坏解析", async () => {
    const missing = parseState({
      revision: 8,
      renderer: {},
      diagnostics: { logs: {} },
    });
    // frame_time_ms 未上报时为 null，计数回退 0，summary 未下发时为 null。
    expect(missing?.renderer.frame_time_ms).toBeNull();
    expect(missing?.renderer.large_frame_deltas).toBe(0);
    expect(missing?.diagnostics.logs.summary).toBeNull();
  });

  it("日志聚合摘要完整维度经分组键按计数展示", async () => {
    const parsed = parseState({
      revision: 9,
      diagnostics: {
        logs: {
          status: "available",
          count: 10,
          summary: {
            total: 10,
            warning_plus: 2,
            by_level: { INFO: 8, WARNING: 2 },
            by_logger: { "app.core": 10 },
            by_event: { "session.start": 9, "frame.large": 1 },
          },
          records: [{ level: "INFO", logger: "app.core", event: "session.start" }],
        },
      },
    });
    const summary = parsed?.diagnostics.logs.summary;
    if (!summary) throw new Error("logs.summary 未解析");
    expect(summary.total).toBe(10);
    expect(summary.warning_plus).toBe(2);
    expect(summary.by_level).toEqual({ INFO: 8, WARNING: 2 });
    expect(summary.by_event["session.start"]).toBe(9);
  });
});
