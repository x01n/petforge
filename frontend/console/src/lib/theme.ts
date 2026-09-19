/**
 * MD3 Web 主题令牌契约。
 *
 * 本模块是 Qt 侧（src/gui/qt6/md3.py 的 md3_web_tokens）与
 * src/gui/web/control_surface.py（_PUBLIC_THEME_KEYS / _PUBLIC_THEME_TOKEN /
 * _PUBLIC_THEME_VALUE / _safe_theme_tokens）在 Web 侧的唯一消费镜像。
 * 改动时必须与 Python 侧保持逐字符一致，禁止单侧漂移。
 *
 * 投影不变量：控制面 HTML 会把同一组已投影主题嵌入初始状态，前端必须
 * 按同一套白名单再次过滤，并把全部合法令牌落地为 CSS 变量。
 * 1. 白名单 = 29 个 --md3-color-<kebab> 颜色键 + 19 个非颜色键，
 *    另含 --md3-seed（Python 集合成员；仅种子非空时出现，按原名注入）；
 * 2. --md3-color-scheme 只驱动 documentElement 的 dark 类，不注入变量；
 * 3. 29 个颜色键全部落地：命中别名表的使用 Tailwind 语义变量名，
 *    未命中别名的以原 kebab 名定义，任何颜色角色都不会被静默丢弃；
 * 4. 其余非颜色令牌（radius/motion/font/density/spacing/height 等）
 *    按原 kebab 名注入同名 CSS 变量，让 Web 渲染尽量接近 Qt 画布。
 */

/** MD3ColorScheme 的 29 个角色字段的 kebab 拼写（md3.py 精确全集）。 */
export const MD3_COLOR_ROLES = [
  "primary",
  "on-primary",
  "primary-container",
  "on-primary-container",
  "secondary",
  "on-secondary",
  "secondary-container",
  "on-secondary-container",
  "tertiary",
  "on-tertiary",
  "error",
  "on-error",
  "error-container",
  "on-error-container",
  "surface",
  "on-surface",
  "surface-variant",
  "on-surface-variant",
  "surface-container-low",
  "surface-container",
  "surface-container-high",
  "outline",
  "outline-variant",
  "inverse-surface",
  "inverse-on-surface",
  "success",
  "on-success",
  "warning",
  "on-warning",
] as const;

/** 29 个 --md3-color-<kebab> 颜色令牌的精确拼写全集。 */
export const PUBLIC_THEME_COLORS = Object.freeze(
  MD3_COLOR_ROLES.map((role) => `--md3-color-${role}`),
);

/**
 * 19 个非颜色令牌的精确拼写全集，顺序与 md3_web_tokens 的
 * tokens.update(...) 字面逐字符一致（含 --md3-color-scheme）。
 */
export const PUBLIC_THEME_METRICS = [
  "--md3-color-scheme",
  "--md3-font-family",
  "--md3-font-family-monospace",
  "--md3-font-size-body",
  "--md3-font-size-label",
  "--md3-font-size-title",
  "--md3-font-size-headline",
  "--md3-radius-small",
  "--md3-radius-medium",
  "--md3-radius-large",
  "--md3-radius-extra-large",
  "--md3-control-height",
  "--md3-compact-height",
  "--md3-density",
  "--md3-spacing",
  "--md3-motion-short",
  "--md3-motion-medium",
  "--md3-motion-long",
  "--md3-reduced-motion",
] as const;

/**
 * Python _PUBLIC_THEME_KEYS 与 md3_web_tokens 输出的差集成员：
 * --md3-seed 只在主题挂起种子时出现，但仍属白名单集合（逐字符镜像要求）。
 */
export const PUBLIC_THEME_EXTRA_KEYS = Object.freeze(["--md3-seed"] as const);

/** Python 侧 _PUBLIC_THEME_TOKEN 的逐字符镜像。 */
export const PUBLIC_THEME_TOKEN = /^--md3-[a-z0-9-]{1,80}$/;

/**
 * Python 侧 _PUBLIC_THEME_VALUE 的逐字符镜像；只放行可安全写入 CSS 的值。
 * 字符类与 Python 源码逐字符相同，其中 \" 是字面引号（Python 侧 '\\"'
 * 在 raw 字符串中的既有拼写），任一侧改动必须同步另一侧。
 */
export const SAFE_THEME_VALUE = /^[#A-Za-z0-9 ,.()'\"_-]{1,128}$/;

/** color-scheme 令牌的精确拼写；只驱动明暗类，不在变量注入之列。 */
export const THEME_COLOR_SCHEME_TOKEN = "--md3-color-scheme";

interface ThemeWhitelist {
  readonly colors: ReadonlySet<string>;
  readonly metrics: ReadonlySet<string>;
  readonly all: ReadonlySet<string>;
}

function buildThemeWhitelist(): ThemeWhitelist {
  const colors = new Set<string>(PUBLIC_THEME_COLORS);
  const metrics = new Set<string>(PUBLIC_THEME_METRICS);
  const all = new Set<string>([...colors, ...metrics, ...PUBLIC_THEME_EXTRA_KEYS]);
  if (colors.size !== MD3_COLOR_ROLES.length) throw new Error("主题颜色令牌键出现重复拼写");
  if (all.size !== colors.size + metrics.size + PUBLIC_THEME_EXTRA_KEYS.length) {
    throw new Error("主题白名单各分区出现重叠键");
  }
  return { colors, metrics, all };
}

const THEME_WHITELIST = buildThemeWhitelist();

/** 与 Python _PUBLIC_THEME_KEYS 相等的键集合常量（含 --md3-seed）。 */
export const PUBLIC_THEME_KEYS: ReadonlySet<string> = THEME_WHITELIST.all;

/** 29 个颜色令牌键集合。 */
export const PUBLIC_THEME_COLOR_KEYS: ReadonlySet<string> = THEME_WHITELIST.colors;

/** 19 个非颜色令牌键集合。 */
export const PUBLIC_THEME_METRIC_KEYS: ReadonlySet<string> = THEME_WHITELIST.metrics;

/**
 * MD3 颜色令牌（kebab 角色） → 既有 Tailwind 语义变量（15 项）。
 * 值已是完整变量名（含 -- 前缀）；未命中别名的角色由调用方以
 * 原 kebab 名（如 --md3-color-surface-variant）定义同名变量。
 */
export const MD3_COLOR_ALIASES: Readonly<Record<string, string>> = Object.freeze({
  surface: "--background",
  "on-surface": "--foreground",
  "surface-container": "--card",
  "surface-container-low": "--sidebar",
  "surface-container-high": "--popover",
  primary: "--primary",
  "on-primary": "--primary-foreground",
  secondary: "--secondary",
  "on-secondary": "--secondary-foreground",
  "primary-container": "--accent",
  "on-primary-container": "--soft-primary",
  outline: "--muted",
  "outline-variant": "--border",
  "on-surface-variant": "--muted-foreground",
  error: "--destructive",
});

/**
 * 把通过白名单的 --md3-color-* 键转换为 Tailwind 消费变量名：
 * --md3-color-surface → --background；未映射角色返回 null，
 * 调用方必须以原 kebab 名注入，不允许静默丢弃。
 */
export function themePropertyName(raw: string): string | null {
  const token = /^--md3-color-([a-z0-9-]+)$/.exec(raw);
  if (!token) return null;
  return MD3_COLOR_ALIASES[token[1]] ?? null;
}