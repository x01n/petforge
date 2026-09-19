import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  MD3_COLOR_ALIASES,
  MD3_COLOR_ROLES,
  PUBLIC_THEME_COLORS,
  PUBLIC_THEME_KEYS,
  PUBLIC_THEME_METRICS,
  SAFE_THEME_VALUE,
} from "./theme";

/**
 * 主题令牌统一：Qt 下发 MD3 令牌时前端先按与 Python 侧一致的键白名单
 * 过滤，再把 29 个颜色令牌全部落入 CSS 变量（别名或原 kebab 名）、
 * 非颜色令牌按原 kebab 名注入，并按 --md3-color-scheme 同步 dark 类。
 * 手动切换按钮的 localStorage 记忆不在此模块处理。
 */
describe("主题令牌白名单契约", () => {
  it("白名单等于 29 个颜色键 + 19 个非颜色键 + 种子键，含 48 项", () => {
    expect(PUBLIC_THEME_COLORS.length).toBe(29);
    expect(PUBLIC_THEME_METRICS.length).toBe(19);
    expect(PUBLIC_THEME_KEYS.size).toBe(49);
    for (const key of PUBLIC_THEME_COLORS) expect(PUBLIC_THEME_KEYS.has(key)).toBe(true);
    for (const key of PUBLIC_THEME_METRICS) expect(PUBLIC_THEME_KEYS.has(key)).toBe(true);
    expect(PUBLIC_THEME_KEYS.has("--md3-seed")).toBe(true);
  });

  it("文件顶部空 token 数组与 Python 侧投影签名一致", () => {
    for (const key of PUBLIC_THEME_COLORS) expect(key).toMatch(/^--md3-color-[a-z0-9-]+$/);
    for (const key of PUBLIC_THEME_METRICS) expect(key).toMatch(/^--md3-[a-z0-9-]+$/);
    for (const key of PUBLIC_THEME_KEYS) expect(key).toMatch(/^--md3-[a-z0-9-]{1,80}$/);
  });
});

describe("Qt 主题令牌同步", () => {
  const classListAdd = vi.fn();
  const classListRemove = vi.fn();
  const classListToggle = vi.fn();
  const setProperty = vi.fn();

  beforeEach(() => {
    vi.resetModules();
    classListAdd.mockClear();
    classListRemove.mockClear();
    classListToggle.mockClear();
    setProperty.mockClear();
    vi.stubGlobal("document", {
      documentElement: {
        classList: { add: classListAdd, remove: classListRemove, toggle: classListToggle },
        style: { setProperty },
      },
      getElementById: () => null,
    });
    vi.stubGlobal("window", {});
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("暗色令牌写入 CSS 变量并添加 dark 类", async () => {
    const { applyState } = await import("./bridge");
    expect(
      applyState({
        revision: 1,
        theme: {
          "--md3-color-surface": "#202020",
          "--md3-color-on-surface": "#FFFFFF",
          "--md3-color-scheme": "dark",
        },
      }),
    ).toBe(true);
    expect(setProperty).toHaveBeenCalledWith("--background", "#202020");
    expect(setProperty).toHaveBeenCalledWith("--foreground", "#FFFFFF");
    expect(classListAdd).toHaveBeenCalledWith("dark");
    expect(classListRemove).not.toHaveBeenCalled();
  });

  it("相同签名重复应用不重复写入（签名去重）", async () => {
    const { applyState } = await import("./bridge");
    const state = {
      revision: 1,
      theme: { "--md3-color-surface": "#202020", "--md3-color-scheme": "dark" },
    };
    applyState(state);
    const setCalls = setProperty.mock.calls.length;
    const addCalls = classListAdd.mock.calls.length;
    // 主动态变化后 Qt 空闲轮询会推送相同主题的新 revision；令牌部分应跳过。
    applyState({ ...state, revision: 2 });
    expect(setProperty).toHaveBeenCalledTimes(setCalls);
    expect(classListAdd).toHaveBeenCalledTimes(addCalls);
  });

  it("浅色令牌移除 dark 类", async () => {
    const { applyState } = await import("./bridge");
    applyState({
      revision: 1,
      theme: { "--md3-color-surface": "#F3F3F3", "--md3-color-scheme": "light" },
    });
    expect(setProperty).toHaveBeenCalledWith("--background", "#F3F3F3");
    expect(classListRemove).toHaveBeenCalledWith("dark");
    expect(classListAdd).not.toHaveBeenCalled();
  });

  it("未下发 color-scheme 时不动明暗类", async () => {
    const { applyState } = await import("./bridge");
    applyState({ revision: 1, theme: { "--md3-color-surface": "#202020" } });
    expect(setProperty).toHaveBeenCalledWith("--background", "#202020");
    expect(classListAdd).not.toHaveBeenCalled();
    expect(classListRemove).not.toHaveBeenCalled();
  });

  it("localStorage 无效主题记忆回退 light：宿主令牌仍驱动明暗类", async () => {
    const storage: Record<string, string> = { "meapet-console-theme": "foo" };
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage[key] ?? null,
      setItem: (key: string, value: string) => { storage[key] = value; },
      removeItem: (key: string) => { delete storage[key]; },
    });
    const { applyState, loadStoredTheme } = await import("./bridge");
    // 非法记忆不通过白名单校验，App 侧状态回退 light；宿主令牌仍生效。
    expect(loadStoredTheme()).toBeNull();
    applyState({
      revision: 1,
      theme: { "--md3-color-surface": "#F3F3F3", "--md3-color-scheme": "dark" },
    });
    expect(classListAdd).toHaveBeenCalledWith("dark");
  });

  it("用户接管主题后宿主令牌不再推动明暗类", async () => {
    const storage: Record<string, string> = {};
    vi.stubGlobal("window", {
      localStorage: {
        getItem: (key: string) => storage[key] ?? null,
        setItem: (key: string, value: string) => { storage[key] = value; },
        removeItem: (key: string) => { delete storage[key]; },
      },
    });
    const { applyState, applyUserTheme } = await import("./bridge");
    applyState({
      revision: 1,
      theme: { "--md3-color-surface": "#202020", "--md3-color-scheme": "dark" },
    });
    expect(classListAdd).toHaveBeenCalledWith("dark");
    classListAdd.mockClear();
    // 用户点击切换后写接管标志；同签名令牌再推送也不覆盖（签名去重 + 豁免双保险）。
    applyUserTheme("light");
    expect(classListToggle).toHaveBeenCalledWith("dark", false);
    applyState({
      revision: 2,
      theme: { "--md3-color-surface": "#101010", "--md3-color-scheme": "dark" },
    });
    expect(classListAdd).not.toHaveBeenCalled();
  });

  it("非法令牌值被跳过，不写入 CSS", async () => {
    const { applyState } = await import("./bridge");
    applyState({
      revision: 1,
      theme: { "--md3-color-surface": "@@@", "--md3-color-scheme": "dark" },
    });
    expect(setProperty).not.toHaveBeenCalled();
    // color-scheme 仍由安全白名单放行。
    expect(classListAdd).toHaveBeenCalledWith("dark");
  });

  it("29 个颜色角色全部注入为 CSS 变量，未映射者保留原 kebab 名", async () => {
    const { applyState } = await import("./bridge");
    const colors: Record<string, string> = {};
    MD3_COLOR_ROLES.forEach((role, index) => {
      colors[`--md3-color-${role}`] = `#${index.toString(16).padStart(6, "0")}`;
    });
    applyState({ revision: 1, theme: colors } as unknown as Parameters<typeof applyState>[0]);
    expect(setProperty).toHaveBeenCalledTimes(29);
    const keys = setProperty.mock.calls.map((call: unknown[]) => call[0]);
    MD3_COLOR_ROLES.forEach((role, index) => {
      const value = `#${index.toString(16).padStart(6, "0")}`;
      const variable = MD3_COLOR_ALIASES[role] ?? `--md3-color-${role}`;
      expect(setProperty).toHaveBeenCalledWith(variable, value);
    });
    // 未映射的颜色角色（surface_variant 等）以原 kebab 名注入。
    expect(keys).toContain("--md3-color-surface-variant");
  });

  it("非颜色令牌白名单通过且以同名变量注入（radius/motion/font/density/spacing）", async () => {
    const { applyState } = await import("./bridge");
    applyState({
      revision: 1,
      theme: {
        "--md3-radius-small": "4px",
        "--md3-radius-medium": "8px",
        "--md3-radius-large": "12px",
        "--md3-radius-extra-large": "16px",
        "--md3-motion-short": "120ms",
        "--md3-motion-medium": "240ms",
        "--md3-motion-long": "480ms",
        "--md3-font-family": "'Microsoft YaHei', sans-serif",
        "--md3-font-family-monospace": "Consolas",
        "--md3-font-size-body": "14px",
        "--md3-font-size-label": "12px",
        "--md3-font-size-title": "16px",
        "--md3-font-size-headline": "22px",
        "--md3-density": "-1",
        "--md3-spacing": "8px",
        "--md3-control-height": "36px",
        "--md3-compact-height": "30px",
        "--md3-reduced-motion": "1",
        "--md3-color-scheme": "dark",
      },
    });
    for (const key of PUBLIC_THEME_METRICS) {
      if (key === "--md3-color-scheme") continue;
      expect(setProperty).toHaveBeenCalledWith(key, expect.anything());
    }
    expect(setProperty).not.toHaveBeenCalledWith("--md3-color-scheme", expect.anything());
    expect(classListAdd).toHaveBeenCalledWith("dark");
    expect(setProperty).toHaveBeenCalledTimes(18);
  });

  it("seed 令牌按原名注入", async () => {
    const { applyState } = await import("./bridge");
    applyState({ revision: 1, theme: { "--md3-seed": "#6750A4" } });
    expect(setProperty).toHaveBeenCalledWith("--md3-seed", "#6750A4");
  });

  it("白名单外的未知键被拒绝", async () => {
    const { applyState } = await import("./bridge");
    applyState({
      revision: 1,
      theme: {
        "--md3-color-unknown": "#FFFFFF",
        "--md3-evil": "red",
        "--md3-color-error": "#FF0000",
      },
    });
    expect(setProperty).toHaveBeenCalledTimes(1);
    expect(setProperty).toHaveBeenCalledWith("--destructive", "#FF0000");
    expect(setProperty).not.toHaveBeenCalledWith(expect.stringContaining("unknown"), expect.anything());
  });

  it("非法值（;、url(、引号外注入字符）被拒绝", async () => {
    const { applyState } = await import("./bridge");
    applyState({
      revision: 1,
      theme: {
        "--md3-color-primary": "#123456; display: none",
        "--md3-color-secondary": "url(file:///etc/passwd)",
        "--md3-color-tertiary": "#ABCDEF\"; } body { display:none",
        "--md3-radius-large": "24px",
      },
    });
    expect(setProperty).toHaveBeenCalledTimes(1);
    expect(setProperty).toHaveBeenCalledWith("--md3-radius-large", "24px");
  });

  it("缺 token 时保留默认主题，不破坏页面", async () => {
    const { applyState } = await import("./bridge");
    expect(applyState({ revision: 1 })).toBe(true);
    expect(setProperty).not.toHaveBeenCalled();
    expect(classListAdd).not.toHaveBeenCalled();
    expect(classListRemove).not.toHaveBeenCalled();
  });
});

describe("SAFE_THEME_VALUE 镜像", () => {
  it("与 Python 侧正则逐字符一致，放行常用字体与尺寸单位", () => {
    expect(SAFE_THEME_VALUE.test("#6750A4")).toBe(true);
    expect(SAFE_THEME_VALUE.test("rgb(103, 80, 164)")).toBe(true);
    expect(SAFE_THEME_VALUE.test("'Microsoft YaHei', sans-serif")).toBe(true);
    expect(SAFE_THEME_VALUE.test("14px")).toBe(true);
    expect(SAFE_THEME_VALUE.test("cubic-bezier(0.2, 0, 0, 1)")).toBe(true);
    expect(SAFE_THEME_VALUE.test("-1")).toBe(true);
  });
  it("拒绝注入字符、超长与空串", () => {
    expect(SAFE_THEME_VALUE.test("red; background:url(x)")).toBe(false);
    expect(SAFE_THEME_VALUE.test("url(data:image/svg+xml)")).toBe(false);
    expect(SAFE_THEME_VALUE.test('1px"}')).toBe(false);
    expect(SAFE_THEME_VALUE.test("x".repeat(129))).toBe(false);
    expect(SAFE_THEME_VALUE.test("")).toBe(false);
  });
});