/**
 * 窄屏侧边栏宽度断点契约。
 *
 * 阈值选 760px 而不是 1000px 的理由：
 * - CSS 内容断点（1080/980/850/700/460）已自行降级网格，980-1080 区间
 *   主内容只是适度压缩，导航可用性优先，过早折叠反而削弱操作效率；
 * - 1080 断点起 models/pet/settings 已单列、850 断点概要页单列，说明
 *   主内容宽度压力集中在 850 以下；
 * - 侧栏 208px 在 760-850 区间让内容区仅剩 552-642px 而网格仍是双列，
 *   折叠为 56px 后内容区立即获得约 152px，压力最大的一段被精确覆盖；
 * - 700px 内容断点及以下保持 56px 图标栏（不做完全隐藏），桌面嵌入式
 *   客户端的导航入口始终可用。
 */
export const NAV_SIDEBAR_WIDTH = 208;
export const NAV_RAIL_WIDTH = 56;
export const NAV_COLLAPSE_MAX_WIDTH = 760;
export const NAV_ROOT_ATTR = "data-collapsed-nav";

export type NavShape = "flat" | "rail" | "full";

/**
 * 由视口宽度推导侧边栏形态，规则：
 * - width >= 760：完整侧边栏（208px）；
 * - 360 <= width < 760：窄图标栏（56px），700/460 等 CSS 断点规则
 *   仍按媒体查询叠加生效；
 * - width < 360：隐藏侧栏（嵌 Qt 窗口把内容留给主区）。
 */
export function navShapeForWidth(width: number): NavShape {
  if (width < 360) return "flat";
  if (width < NAV_COLLAPSE_MAX_WIDTH) return "rail";
  return "full";
}

/**
 * 将形态同步到 html 根元素（初始无属性时 CSS 回退为完整侧栏，
 * 避免客户端首帧闪烁）：
 * - "full"：完整侧边栏（208px）；
 * - "rail"：窄图标栏（56px），可临时展开为完整导航；
 * - "flat"：<360px 隐藏侧栏，内容区顶满。
 * 空 root（测试 stub）时幂等返回。
 */
export function syncNavShapeAttr(root: HTMLElement | null, shape: NavShape): void {
  if (!root) return;
  root.setAttribute(NAV_ROOT_ATTR, shape);
}