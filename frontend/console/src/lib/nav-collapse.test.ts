import { describe, expect, it } from "vitest";
import {
  NAV_COLLAPSE_MAX_WIDTH,
  NAV_ROOT_ATTR,
  NavShape,
  navShapeForWidth,
  syncNavShapeAttr,
} from "./nav-collapse";

/**
 * 侧边栏宽度断点契约：>=760 完整侧边栏（208px）、360-759 窄图标栏
 * （56px）、<360 隐藏侧栏；属性同步为空 root 时幂等。
 */
describe("侧边栏宽度断点契约", () => {
  it("窗口宽度 760 及以上为完整侧边栏", () => {
    expect(navShapeForWidth(760)).toBe("full");
    expect(navShapeForWidth(1200)).toBe("full");
  });

  it("窗口宽度 360 到 759 之间为窄图标栏", () => {
    for (const width of [759, NAV_COLLAPSE_MAX_WIDTH - 1, 500, 360]) {
      expect(navShapeForWidth(width)).toBe("rail");
    }
  });

  it("宽度低于 360 时隐藏侧栏", () => {
    expect(navShapeForWidth(359)).toBe("flat");
    expect(navShapeForWidth(0)).toBe("flat");
  });
});

describe("侧边栏形态属性同步", () => {
  const attributes: Record<string, string> = {};
  const root = {
    setAttribute: (name: string, value: string) => { attributes[name] = value; },
    removeAttribute: (name: string) => { delete attributes[name]; },
  } as unknown as HTMLElement;

  it("三种形态写入根属性，初始无属性时 CSS 回退为完整侧栏", () => {
    syncNavShapeAttr(root, "rail");
    expect(attributes[NAV_ROOT_ATTR]).toBe("rail");
    syncNavShapeAttr(root, "full");
    expect(attributes[NAV_ROOT_ATTR]).toBe("full");
    syncNavShapeAttr(root, "flat");
    expect(attributes[NAV_ROOT_ATTR]).toBe("flat");
  });

  it("空 root 时幂等返回", () => {
    expect(() => syncNavShapeAttr(null, "rail" as NavShape)).not.toThrow();
  });
});