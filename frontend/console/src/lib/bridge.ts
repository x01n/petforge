import { useCallback, useRef, useSyncExternalStore } from "react";
import { parseResult, parseState, type ActionPayload, type ActionResult, type Bridge, type ControlState } from "./contract";
import {
  MD3_COLOR_ALIASES,
  PUBLIC_THEME_KEYS,
  SAFE_THEME_VALUE,
  THEME_COLOR_SCHEME_TOKEN,
} from "./theme";

declare global {
  interface Window {
    qt?: { webChannelTransport: unknown };
    QWebChannel?: new (transport: unknown, callback: (channel: { objects: Record<string, unknown> }) => void) => unknown;
    __MEAPET_INITIAL_STATE__?: unknown;
    meapetControlSurface: { setState(raw: unknown): boolean; onAction: ((kind: string, payload: ActionPayload) => unknown) | null };
    env?: Record<string, string>;
  }
}

type Snapshot = { state: ControlState | null; connected: boolean; pending: ReadonlySet<string>; result: ActionResult | null };
let snapshot: Snapshot = { state: null, connected: false, pending: new Set(), result: null };
let bridge: Bridge | null = null;
const listeners = new Set<() => void>();
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };
function update(next: Partial<Snapshot>) {
  // Qt 状态轮询可能在页面不可用期间重复投递相同连接状态；
  // 没有字段变化时不通知 React，避免所有控制项被无效重渲染。
  const changed = (Object.keys(next) as Array<keyof Snapshot>).some(
    (key) => !Object.is(snapshot[key], next[key]),
  );
  if (!changed) return;
  snapshot = { ...snapshot, ...next };
  listeners.forEach((listener) => listener());
}

function decodeState(raw: unknown): unknown {
  if (typeof raw !== "string") return raw;
  try { return JSON.parse(raw); } catch { return null; }
}

function readRevision(raw: unknown): number | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const value = (raw as Record<string, unknown>).revision;
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : null;
}

export function applyState(raw: unknown): boolean {
  const decoded = decodeState(raw);
  const revision = readRevision(decoded);
  // 宿主只在业务状态变化时推进 revision。重复快照在进入 Zod 校验前直接丢弃，
  // 避免 Qt 的空闲轮询反复解析完整状态并触发 React 订阅者。
  if (revision !== null && snapshot.state && revision <= snapshot.state.revision) {
    return false;
  }
  const state = parseState(decoded);
  if (!state || (snapshot.state && state.revision < snapshot.state.revision)) return false;
  // 主题令牌：Qt 侧按 MD3 配置推送，只有存在差异时更新。
  // 前端以 CSS 变量桥接消费，缺省时保持编译期默认主题。
  applyThemeTokens(state);
  update({ state });
  return true;
}

function safeThemeValue(raw: unknown): string | null {
  if (raw === null || raw === undefined) return null;
  const value = String(raw);
  return SAFE_THEME_VALUE.test(value) ? value : null;
}

let lastThemeSignature = "";
/**
 * 把 Qt 下发的 MD3 主题令牌按白名单落地为 CSS 变量：
 * 1. 键白名单与 Python _PUBLIC_THEME_KEYS 等同，未入名单的键拒绝；
 * 2. --md3-color-* 转成 --color-<md3 角色> 消费标记，未映射颜色角色
 *    一律以原 kebab 名字定义变量，不允许静默丢弃；
 * 3. 非颜色令牌按原 kebab 名注入同名 CSS 变量（让 Web 渲染接近 Qt）；
 * 4. --md3-color-scheme 只驱动 documentElement 的 dark 类，不注入变量。
 */
function applyThemeTokens(state: ControlState): void {
  const theme = state.theme as Record<string, string> | undefined;
  if (!theme || typeof document === "undefined") return;
  const entries = Object.entries(theme);
  if (!entries.length) return;
  const signature = entries
    .map(([key, value]) => `${key}:${value}`)
    .sort()
    .join(";");
  // 签名去重：Qt 空闲轮询重复推送同一主题时不重复写入 CSS 变量，
  // 也保证用户手动切换明暗后不会被相同的令牌重复推送覆盖。
  if (signature === lastThemeSignature) return;
  const root = document?.documentElement;
  if (!root?.style) return;
  lastThemeSignature = signature;
  for (const [rawKey, rawValue] of entries) {
    const value = safeThemeValue(rawValue);
    if (value === null) continue;
    if (!PUBLIC_THEME_KEYS.has(rawKey)) continue;
    if (rawKey === THEME_COLOR_SCHEME_TOKEN) continue;
    if (rawKey.startsWith("--md3-color-")) {
      const role = rawKey.slice("--md3-color-".length);
      const variable = MD3_COLOR_ALIASES[role] ?? rawKey;
      root.style.setProperty(variable, value);
      continue;
    }
    root.style.setProperty(rawKey, value);
  }
  // Qt 场景下原生令牌优先：color-scheme 明确时由令牌驱动明暗类，
  // 避免与编译期 .dark 块冲突；未下发该令牌时保持 App 手动切换逻辑。
  // 用户点击过明暗切换后令牌不再覆盖用户选择（矩形判定豁免）。
  const scheme = safeThemeValue(theme[THEME_COLOR_SCHEME_TOKEN]);
  if (scheme && !themeHandledByUser()) {
    if (scheme === "dark") root.classList.add("dark");
    else if (scheme === "light") root.classList.remove("dark");
  }
}

type SnapshotSelector<T> = (value: Snapshot) => T;
type SnapshotEquality<T> = (previous: T, next: T) => boolean;

const identitySelector: SnapshotSelector<Snapshot> = (value) => value;

/**
 * 只订阅控制台需要的快照片段，避免流式文本更新时重渲染所有操作控件。
 * 选择器返回已有引用或基础值时，默认的 Object.is 即可完成稳定比较；
 * 对派生对象可传入比较函数复用上一次结果。
 */
export function useControlSurface<T = Snapshot>(
  selector: SnapshotSelector<T> = identitySelector as SnapshotSelector<T>,
  equality: SnapshotEquality<T> = Object.is,
): T {
  const selectorRef = useRef(selector);
  const equalityRef = useRef(equality);
  const cacheRef = useRef<{ source: Snapshot; value: T; selector: SnapshotSelector<T> } | null>(null);
  selectorRef.current = selector;
  equalityRef.current = equality;
  const getSelectedSnapshot = useCallback(() => {
    const source = snapshot;
    const cached = cacheRef.current;
    const selector = selectorRef.current;
    if (cached?.source === source && cached.selector === selector) return cached.value;
    const next = selector(source);
    if (cached && equalityRef.current(cached.value, next)) {
      cacheRef.current = { source, value: cached.value, selector };
      return cached.value;
    }
    cacheRef.current = { source, value: next, selector };
    return next;
  }, []);
  return useSyncExternalStore(subscribe, getSelectedSnapshot, getSelectedSnapshot);
}

export function refreshState() {
  try { bridge?.getState(applyState); } catch { update({ connected: false }); }
}

function browserLocalStorage(): Pick<Storage, "getItem" | "setItem" | "removeItem"> {
  return window.localStorage;
}

/**
 * 载入并校验 localStorage 的主题记忆，非法值回退 light。
 * 元素注入主题的宿主（经 __MEAPET_INITIAL_STATE__ 或 <script> 的元素）没有
 * localStorage 时返回 null，表示"尚无用户接管主题"。
 */
export function loadStoredTheme(): "light" | "dark" | null {
  try {
    const raw = browserLocalStorage().getItem("meapet-console-theme");
    if (raw === "dark" || raw === "light") return raw;
  } catch { /* localStorage 在嵌入式 Qt 页面可能不可用。 */ }
  return null;
}

const THEME_HANDLED_BY_USER = "meapet-console-theme-user";
/** 用户点击过明暗切换按钮后宿主令牌不再改变明暗类；换页/重载凭标志生效。 */
export function themeHandledByUser(): boolean {
  try { return browserLocalStorage().getItem(THEME_HANDLED_BY_USER) === "1"; } catch { return false; }
}

function setThemeUserFlag(value: boolean) {
  try {
    if (value) browserLocalStorage().setItem(THEME_HANDLED_BY_USER, "1");
    else browserLocalStorage().removeItem(THEME_HANDLED_BY_USER);
  } catch { /* localStorage 在嵌入式 Qt 页面可能不可用。 */ }
}

/**
 * 用户手动切换明暗主题：写记忆 + 用户接管标志 + 同步暗色类。
 * 成功后宿主令牌不再覆盖用户选择的明暗（applyThemeTokens 据此豁免）。
 */
export function applyUserTheme(theme: "light" | "dark"): void {
  setThemeUserFlag(true);
  try { browserLocalStorage().setItem("meapet-console-theme", theme); } catch { /* 忽略，水合已保证。 */ }
  document.documentElement.classList.toggle("dark", theme === "dark");
}

export function invoke(kind: string, payload: ActionPayload = {}): Promise<ActionResult> {
  const key = kind + JSON.stringify(payload);
  if (snapshot.pending.has(key)) return Promise.resolve({ status: "pending", message: "操作正在处理中" });
  const localAction = window.meapetControlSurface?.onAction;
  if (!bridge && !localAction) {
    const result = { status: "unavailable", message: "未连接桌宠，请从桌宠托盘打开控制台" };
    update({ result });
    return Promise.resolve(result);
  }
  update({ pending: new Set([...snapshot.pending, key]) });
  return new Promise((resolve) => {
    let settled = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const finish = (raw: unknown) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      const result = parseResult(raw);
      const pending = new Set(snapshot.pending);
      pending.delete(key);
      update({ pending, result });
      refreshState();
      resolve(result);
    };
    timeout = setTimeout(() => finish({ status: "timeout", message: "操作响应超时，请检查桌宠后重试" }), 8000);
    try {
      if (bridge) bridge.invoke(kind, JSON.stringify(payload), finish);
      else Promise.resolve(localAction!(kind, payload)).then(finish, () => finish({ status: "unavailable", message: "操作通道暂时不可用" }));
    } catch { finish({ status: "unavailable", message: "操作通道暂时不可用" }); }
  });
}

export function initializeBridge() {
  window.meapetControlSurface = { setState: applyState, onAction: null };
  if (window.__MEAPET_INITIAL_STATE__) applyState(window.__MEAPET_INITIAL_STATE__);
  const initial = document.getElementById("meapet-initial-state")?.textContent;
  if (initial) applyState(initial);
  if (!window.qt?.webChannelTransport) return;
  const connect = () => {
    if (!window.QWebChannel || !window.qt) return;
    try {
      new window.QWebChannel(window.qt.webChannelTransport, (channel) => {
        const value = channel.objects.meapetControlSurfaceBridge as Partial<Bridge> | undefined;
        if (!value || typeof value.getState !== "function" || typeof value.invoke !== "function") return;
        bridge = value as Bridge;
        update({ connected: true });
        refreshState();
      });
    } catch { update({ connected: false }); }
  };
  if (window.QWebChannel) connect();
  else {
    const script = document.createElement("script");
    script.src = "qrc:///qtwebchannel/qwebchannel.js";
    script.onload = connect;
    document.head.appendChild(script);
  }
}
