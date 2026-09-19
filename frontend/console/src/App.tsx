import { memo, useCallback, useEffect, useRef, useState } from "react";
import { Activity, ArrowUpRight, AudioLines, BrainCircuit, Cat, ChevronLeft, ChevronRight, Command, FileSliders, LayoutDashboard, MessageCircle, Moon, PawPrint, Settings2, Sun, WifiOff, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ActionButton, Panel, Status } from "@/components/console-controls";
import { Conversation } from "@/components/conversation";
import { ModelSettings } from "@/components/model-settings";
import { PetSettings } from "@/components/pet-settings";
import { ActivitySettings, Configuration, Timeline } from "@/components/activity-settings";
import { useControlSurface, applyUserTheme, loadStoredTheme } from "@/lib/bridge";
import { navShapeForWidth, syncNavShapeAttr } from "@/lib/nav-collapse";
import type { NavShape } from "@/lib/nav-collapse";

const pages = [
  { id: "overview", label: "总览", caption: "和你的桌宠，保持连接。", icon: LayoutDashboard },
  { id: "conversation", label: "对话", caption: "想说的话，从这里开始。", icon: MessageCircle },
  { id: "models", label: "模型与语音", caption: "选择大脑，也选择声音。", icon: BrainCircuit },
  { id: "pet", label: "桌宠与动作", caption: "让桌面上的陪伴更自然。", icon: PawPrint },
  { id: "activity", label: "记忆与活动", caption: "记录每一次互动的进展。", icon: Activity },
  { id: "configuration", label: "配置", caption: "让一切按你的习惯运行。", icon: FileSliders },
] as const;
type Page = typeof pages[number]["id"];

/** 由视口宽度推导侧栏形态（rail=360-759 窄图标栏），含 window stub 保护。 */
function currentNavShape(): NavShape {
  if (typeof window === "undefined") return "full";
  return navShapeForWidth(window.innerWidth);
}

const Sidebar = memo(function Sidebar({ page, connected, navigate, pinned, railMode, onPinnedChange }: { page: Page; connected: boolean; navigate: (page: Page) => void; pinned: boolean; railMode: NavShape; onPinnedChange: (next: boolean) => void }) {
  const asideRef = useRef<HTMLElement>(null);
  useEffect(() => {
    // 图标栏下保持内容区不下塌：悬停即展开完整导航；无 hover 的触屏
    // 设备常驻图标栏，由底部的展开/收起按钮驱动。
    if (pinned || railMode !== "rail") return;
    const aside = asideRef.current;
    if (!aside) return;
    const onEnter = () => onPinnedChange(true);
    aside.addEventListener("mouseenter", onEnter);
    return () => aside.removeEventListener("mouseenter", onEnter);
  }, [pinned, railMode, onPinnedChange]);
  useEffect(() => {
    // Esc 或点击栏外区域把临时展开的完整导航收回为图标栏。
    if (!pinned || railMode !== "rail") return;
    const aside = asideRef.current;
    if (!aside) return;
    const onPointerDown = (event: MouseEvent) => {
      if (event.target instanceof Node && aside.contains(event.target)) return;
      onPinnedChange(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onPinnedChange(false);
        const active = document.activeElement;
        if (active instanceof HTMLElement && aside.contains(active)) active.blur();
      }
    };
    window.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [pinned, railMode, onPinnedChange]);
  const asideLabel = railMode === "rail" ? (pinned ? "导航已临时展开" : "导航图标栏") : "导航";
  return <aside ref={asideRef} className="desktop-sidebar" aria-label={asideLabel}>
    <div className="brand"><div className="brand-mark"><Cat strokeWidth={1.6} /></div><strong>MeaPet<span>控制台</span></strong></div>
    <nav aria-label="控制台导航" className="nav-list">{pages.map((item) => <Button key={item.id} variant="ghost" className={`nav-item ${page === item.id ? "nav-active" : ""}`} aria-current={page === item.id ? "page" : undefined} aria-label={item.label} data-page={item.id} onClick={() => { navigate(item.id); onPinnedChange(false); }}><item.icon /><span>{item.label}</span></Button>)}</nav>
    <div className="sidebar-bottom"><Separator /><div className="sidebar-status"><span className={connected ? "live-dot" : "offline-dot"} /><span>{connected ? "桌宠已连接" : "未连接桌宠"}</span></div><Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" className="nav-toggle" aria-label={pinned ? "收起导航" : "展开导航"} aria-expanded={pinned} data-active={pinned} onClick={() => onPinnedChange(!pinned)}>{pinned ? <ChevronLeft /> : <ChevronRight />}</Button></TooltipTrigger><TooltipContent>{pinned ? "收起导航" : "展开导航"}</TooltipContent></Tooltip></div>
  </aside>;
});

/** 载入时按 localStorage 记忆回退：非法值落到 light，不做任何猜测。 */
function getTheme() {
  return loadStoredTheme() ?? "light";
}

export default function App() {
  const state = useControlSurface((current) => current.state);
  const connected = useControlSurface((current) => current.connected);
  const result = useControlSurface((current) => current.result);
  const [page, setPage] = useState<Page>("overview");
  const [theme, setTheme] = useState(getTheme);
  const [dismissedResult, setDismissedResult] = useState<typeof result>(null);
  const [navPinned, setNavPinned] = useState(false);
  const [railMode, setRailMode] = useState<NavShape>(currentNavShape);
  const [compactHeight, setCompactHeight] = useState(
    typeof window !== "undefined" && window.innerHeight < 620
  );
  useEffect(() => {
    // 折叠由 matchMedia 按 760px 阈值驱动：data-collapsed-nav（"rail"=
    // 窄图标栏 / "full" / "flat"）只落在根元素上，与既有
    // 1080/980/850/700/460 五个 CSS 媒体查询压缩规则叠加生效；离开窄栏
    // 时复位浮层标记，避免残留展开态。
    const root = document.documentElement;
    const railMedia = window.matchMedia("(max-width: 759px)");
    const sync = () => {
      const shape = currentNavShape();
      syncNavShapeAttr(root, shape);
      setRailMode(shape);
      if (shape !== "rail") setNavPinned(false);
    };
    sync();
    railMedia.addEventListener("change", sync);
    return () => railMedia.removeEventListener("change", sync);
  }, []);
  useEffect(() => {
    // nav-pinned 是浮层展开的唯一视觉驱动，仅在窄图标栏生效。
    document.documentElement.classList.toggle("nav-pinned", navPinned && railMode === "rail");
  }, [navPinned, railMode]);
  useEffect(() => {
    // 高度方向没有媒体查询时，ResizeObserver 保持关键操作（发送、
    // 切换、恢复按钮）在任意窗口高度下布局一致。
    const root = document.documentElement;
    const observer = new ResizeObserver(() => {
      setCompactHeight(root.clientHeight < 620);
      root.setAttribute("data-compact-height", String(root.clientHeight < 620));
    });
    observer.observe(root);
    setCompactHeight(root.clientHeight < 620);
    root.setAttribute("data-compact-height", String(root.clientHeight < 620));
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    document.documentElement.classList.toggle("compact", compactHeight);
  }, [compactHeight]);
  const current = pages.find((item) => item.id === page)!;
  const activeChannel = state?.model_channels.find((channel) => channel.active);
  useEffect(() => {
    // 桌面 Qt 会话由桥接层按 --md3-color-scheme 驱动 .dark 类（原生令牌
    // 优先）；用户点击过切换按钮后桥接层豁免令牌推动。这里的 toggle 仅
    // 把本地 state 落到 DOM 类，手工选择经 applyUserTheme 持久化。
    document.documentElement.classList.toggle("dark", theme === "dark");
    try { localStorage.setItem("meapet-console-theme", theme); } catch { /* Local storage may be unavailable in embedded Qt pages. */ }
  }, [theme]);
  const navigate = useCallback((next: Page) => setPage(next), []);
  return <TooltipProvider delayDuration={200}><div className="console-shell">
    <Sidebar page={page} connected={connected} navigate={navigate} pinned={navPinned} railMode={railMode} onPinnedChange={setNavPinned} />
    <div className="workspace"><header className="topbar"><div className="topbar-start"><span className="breadcrumb">控制台 <ChevronRight /> <strong>{current.label}</strong></span></div><div className="topbar-actions"><Status good={connected}>{connected ? state?.connection.state || "已连接" : "未连接"}</Status><Separator orientation="vertical" /><Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" aria-label={theme === "dark" ? "切换浅色主题" : "切换深色主题"} onClick={() => { const next = theme === "dark" ? "light" : "dark"; applyUserTheme(next); setTheme(next); }}>{theme === "dark" ? <Sun /> : <Moon />}</Button></TooltipTrigger><TooltipContent>切换明暗主题</TooltipContent></Tooltip><ActionButton kind="open_config" variant="ghost" size="icon" aria-label="打开配置中心"><Settings2 /></ActionButton></div></header>
      <main className="main-content" id="main-content"><div className="page-heading"><div><h1>{current.label}</h1><p>{current.caption}</p></div>{page === "overview" && <ActionButton kind="show_pet" variant="outline"><PawPrint />显示桌宠</ActionButton>}</div>
        {!connected && <div className="connection-banner" role="status"><WifiOff /><div><strong>未连接桌宠</strong><span>界面已就绪，请通过运行中的桌宠打开控制台。</span></div></div>}
        {connected && state?.interaction.model.ready === false && <div className="connection-banner" role="status"><BrainCircuit /><div className="flex-1"><strong>连接一个模型，开始对话</strong><span>{state.interaction.model.message}</span></div><ActionButton kind="configure_model" size="sm">配置模型</ActionButton></div>}
        {result && result !== dismissedResult && <div role="status" className={`result-banner ${["failed", "error", "unavailable", "timeout", "denied", "rejected"].includes(result.status) ? "result-error" : ""}`}><span>{result.message || result.reason || result.detail || result.status_label || "操作已提交"}</span><Button variant="ghost" size="icon" aria-label="关闭操作提示" onClick={() => setDismissedResult(result)}><X /></Button></div>}
        {page === "overview" && <><div className="metrics-grid">{[
          { label: "对话模型", value: activeChannel?.model || (connected ? "尚未连接模型" : "等待连接"), hint: activeChannel?.id || "配置你的模型渠道", icon: BrainCircuit, to: "models" as Page },
          { label: "语音输出", value: state?.tts.profile || (connected ? state?.tts.state || "尚未配置" : "等待连接"), hint: state?.tts.backend || "选择桌宠的声音", icon: AudioLines, to: "models" as Page },
          { label: "桌宠渲染", value: state?.renderer.label || "等待连接", hint: state?.renderer.available ? "角色已就绪" : "等待角色状态", icon: PawPrint, to: "pet" as Page },
        ].map((metric) => <Card className="metric-card" key={metric.label}><CardContent><div className="metric-top"><span>{metric.label}</span><metric.icon /></div><strong title={metric.value}>{metric.value}</strong><div className="metric-footer"><span>{metric.hint}</span><Button size="icon" variant="ghost" aria-label={`管理${metric.label}`} onClick={() => navigate(metric.to)}><ArrowUpRight /></Button></div></CardContent></Card>)}</div><div className="overview-grid"><Conversation state={state} compact /><div className="overview-side-stack space-y-5"><Panel title="随手互动" description="给桌宠一点关注" action={<PawPrint className="section-icon" />}><div className="quick-actions">{state?.parts.length ? state.parts.slice(0, 4).map((part) => <ActionButton key={part.id} kind="pet_part" payload={{ part: part.id }} disabled={!part.enabled} variant="outline"><PawPrint />{part.label}</ActionButton>) : <p className="muted-note">连接后显示角色的互动方式。</p>}</div><Button variant="ghost" className="quick-more" onClick={() => navigate("pet")}>全部动作与窗口控制<ArrowUpRight /></Button></Panel><Timeline state={state} /></div></div></>}
        {page === "conversation" && <Conversation state={state} />}
        {page === "models" && <ModelSettings state={state} />}
        {page === "pet" && <PetSettings state={state} />}
        {page === "activity" && <ActivitySettings state={state} />}
        {page === "configuration" && <Configuration state={state} />}
        <footer className="page-footer"><span><Cat />MeaPet · 桌面陪伴</span><span><Command /> Ctrl + Shift + M 唤起控制台</span></footer>
      </main>
    </div>
  </div></TooltipProvider>;
}
