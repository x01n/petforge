import { useState } from "react";
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Expand, Eye, Hand, Layers, LockKeyhole, MousePointer2, Move, Pin, Play, Scan, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ActionButton, Choice, Detail, Panel, Status } from "./console-controls";
import { invoke, useControlSurface } from "@/lib/bridge";
import type { ControlState } from "@/lib/contract";

function Choreography({ state }: { state: ControlState | null }) {
  const connected = useControlSurface((current) => current.connected);
  const [selected, setSelected] = useState<string[]>([]);
  const [mode, setMode] = useState("sequence");
  const [duration, setDuration] = useState("1.5");
  const [transition, setTransition] = useState("0.2");
  const [loop, setLoop] = useState(false);
  const valid = Number(duration) >= .05 && Number(duration) <= 120 && Number(transition) >= 0 && Number(transition) <= 10;
  const offered = new Set((state?.capabilities.expressions ?? []).map((item) => item.id));
  const available = selected.filter((id) => offered.has(id));
  // 宿主推送后已选项可能不再提供：保留选择并渲染为禁用条目，
  // 避免用户选中项被静默过滤；执行载荷仍只包含当前可用的表达式。
  const unavailable = selected.filter((id) => !offered.has(id));
  return <Dialog><DialogTrigger asChild><Button variant="outline" size="sm"><Layers />表情编排</Button></DialogTrigger><DialogContent><DialogHeader><DialogTitle>让表情连贯起来</DialogTitle><DialogDescription>选择表情与过渡时间，顺序播放或混合执行。</DialogDescription></DialogHeader>
    <div className="flex flex-wrap gap-2">{state?.capabilities.expressions.map((expression) => <label className="choice-pill" key={expression.id}><Checkbox checked={available.includes(expression.id)} onCheckedChange={(checked) => setSelected((current) => checked ? [...current, expression.id] : current.filter((id) => id !== expression.id))} />{expression.label}</label>)}
      {unavailable.map((id) => <label className="choice-pill choice-pill-missing" key={id} title="宿主当前未提供该表达式"><Checkbox checked disabled onCheckedChange={() => setSelected((current) => current.filter((item) => item !== id))} />{id}<span className="text-xs opacity-70">（当前不可用）</span></label>)}
    </div>
    <div className="field-stack"><label>执行方式</label><Select value={mode} onValueChange={setMode}><SelectTrigger aria-label="表情执行方式"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="sequence">顺序播放</SelectItem><SelectItem value="blend">混合表情</SelectItem></SelectContent></Select></div>
    <div className="grid grid-cols-2 gap-4"><div className="field-stack"><label htmlFor="expression-duration">持续时间（秒）</label><Input id="expression-duration" type="number" min="0.05" max="120" step="0.1" value={duration} onChange={(event) => setDuration(event.target.value)} /></div><div className="field-stack"><label htmlFor="expression-transition">过渡时间（秒）</label><Input id="expression-transition" type="number" min="0" max="10" step="0.1" value={transition} onChange={(event) => setTransition(event.target.value)} /></div></div>
    <label className="toggle-row"><span>循环播放</span><Switch checked={loop} onCheckedChange={setLoop} /></label>
    {!valid && <p className="text-sm text-destructive">持续时间为 0.05–120 秒，过渡时间为 0–10 秒。</p>}
    <ActionButton kind="expression_request" disabled={!connected || !available.length || !valid} payload={{ mode, loop, expressions: available.map((name) => ({ name, weight: 1, duration_seconds: Number(duration), transition_seconds: Number(transition), parameters: {} })) }}><Play />执行表情编排</ActionButton>
  </DialogContent></Dialog>;
}

export function PetSettings({ state }: { state: ControlState | null }) {
  const connected = useControlSurface((current) => current.connected);
  const pending = useControlSurface((current) => current.pending);
  const switches = [
    { kind: "toggle_window_lock", label: "锁定桌宠位置", detail: "避免拖拽时意外移动", icon: LockKeyhole, checked: state?.window.locked },
    { kind: "toggle_always_on_top", label: "窗口始终置顶", detail: "让桌宠陪伴在其他窗口之上", icon: Pin, checked: state?.window.always_on_top },
    { kind: "toggle_click_through", label: "鼠标点击穿透", detail: "鼠标可以操作桌宠后面的窗口", icon: MousePointer2, checked: state?.window.click_through },
  ];
  return <div className="pet-layout"><div className="space-y-5"><Panel title="桌宠与渲染" description="选择角色、渲染方式与显示尺寸" action={<Status good={state?.renderer.available}>{state?.renderer.label || "等待渲染器"}</Status>}>
    <div className="field-stack"><label>当前 Live2D 模型</label><Choice label="选择 Live2D 模型" value={state?.renderer.model} options={(state?.renderer.models ?? []).map((model) => ({ value: model, label: model }))} kind="select_renderer_model" field="model" /></div>
    <div className="field-stack"><label>切换渲染方式</label><Choice label="选择渲染方式" kind="select_renderer_backend" field="backend" options={[{ value: "auto", label: "自动选择" }, { value: "web_live2d", label: "Web Live2D" }, { value: "opengl", label: "OpenGL Live2D" }, { value: "vulkan", label: "Vulkan 渲染" }, { value: "sprite", label: "精灵（显式）" }]} /></div>
    <div className="details"><Detail label="模型就绪" value={state?.renderer.ready ? "已就绪" : state?.renderer.lifecycle_state || "未就绪"} /><Detail label="实际图形 API" value={state?.renderer.actual_api || "尚未回读"} /><Detail label="目标帧率" value={state?.renderer.frame_rate ? String(Math.round(state.renderer.frame_rate)) + " FPS" : "宿主默认"} /><Detail label="几何审计" value={state?.renderer.geometry_audit_hz ? String(Math.round(state.renderer.geometry_audit_hz)) + " Hz" : "宿主默认"} /><Detail label="已绘制 / 节流帧" value={state ? String(Math.round(state.renderer.rendered_frames)) + " / " + String(Math.round(state.renderer.throttled_frames)) : "尚未采样"} /><Detail label="几何审计次数" value={state ? String(Math.round(state.renderer.geometry_audits)) : "尚未采样"} /><Detail label="几何缓存命中 / 未命中" value={state ? String(Math.round(state.renderer.geometry_cache_hits)) + " / " + String(Math.round(state.renderer.geometry_cache_misses)) : "尚未采样"} /><Detail label="平均帧耗时" value={state ? state.renderer.frame_time_ms != null ? state.renderer.frame_time_ms.toFixed(1) + " ms" : "暂无样本" : "尚未采样"} /><Detail label="大帧钳制计数" value={state ? String(Math.round(state.renderer.large_frame_deltas)) : "尚未采样"} /></div>
    <p className="muted-note">{state?.renderer.message || "可用性由桌宠检测；切换结果会显示在操作回执中。"}</p>
    <div className="field-stack"><label>显示尺寸</label><div className="segmented-actions">{[{ preset: "small", label: "小巧" }, { preset: "standard", label: "标准" }, { preset: "large", label: "大号" }].map((item) => <ActionButton key={item.preset} kind="set_display_size" payload={{ preset: item.preset }} variant="outline"><Expand />{item.label}</ActionButton>)}</div></div>
    <div className="flex flex-wrap gap-2"><ActionButton kind="show_pet" variant="outline"><Eye />显示桌宠</ActionButton><ActionButton kind="toggle_visibility" variant="outline">切换显示 / 隐藏</ActionButton></div>
  </Panel><Panel title="表情与互动" description="触发当前角色支持的动作" action={<Choreography state={state} />}>
    <div className="field-stack"><label><Sparkles />表情</label><div className="flex flex-wrap gap-2">{state?.capabilities.expressions.map((item) => <ActionButton key={item.id} kind="expression" payload={{ name: item.id }} variant="outline" size="sm">{item.label}</ActionButton>)}{!state && <p className="muted-note">连接后显示角色的表情列表</p>}</div></div>
    <div className="field-stack"><label><Play />动作</label><div className="flex flex-wrap gap-2">{state?.capabilities.motions.map((item) => <ActionButton key={item.id} kind="motion" payload={{ name: item.id }} variant="outline" size="sm">{item.label}</ActionButton>)}</div></div>
    <div className="field-stack"><label><Hand />部位互动</label><div className="flex flex-wrap gap-2">{state?.parts.map((part) => <ActionButton key={part.id} kind="pet_part" payload={{ part: part.id }} disabled={!part.enabled} variant="secondary" size="sm">{part.label}</ActionButton>)}</div></div>
    {(state?.affection.tier || state?.affection.mood) && <div className="details"><Detail label="与你的关系" value={`${state.affection.tier} · ${state.affection.current}`} />{state?.affection.mood && <Detail label="当前心情" value={state.affection.mood} />}</div>}
  </Panel></div><div className="space-y-5"><Panel title="位置与窗口" description="把桌宠放到舒服的位置" action={<Move className="section-icon" />}>
    <div className="direction-pad"><ActionButton className="direction-up" kind="nudge_pet" payload={{ direction: "up" }} variant="outline" size="icon" aria-label="向上移动"><ArrowUp /></ActionButton><ActionButton className="direction-left" kind="nudge_pet" payload={{ direction: "left" }} variant="outline" size="icon" aria-label="向左移动"><ArrowLeft /></ActionButton><ActionButton className="direction-center" kind="center_pet" variant="secondary" size="icon" aria-label="居中桌宠"><Scan /></ActionButton><ActionButton className="direction-right" kind="nudge_pet" payload={{ direction: "right" }} variant="outline" size="icon" aria-label="向右移动"><ArrowRight /></ActionButton><ActionButton className="direction-down" kind="nudge_pet" payload={{ direction: "down" }} variant="outline" size="icon" aria-label="向下移动"><ArrowDown /></ActionButton></div>
    <Detail label="桌面位置" value={state?.window.position ? `${state.window.position.x}, ${state.window.position.y}` : "尚未上报"} />
    {switches.map((item) => <label className="toggle-row" key={item.kind}><item.icon /><span className="flex-1"><strong>{item.label}</strong><small>{item.detail}</small></span><Switch data-action={item.kind} aria-label={item.label} checked={Boolean(item.checked)} disabled={!connected || [...pending].some((key) => key.startsWith(item.kind + "{"))} onCheckedChange={() => { void invoke(item.kind); }} /></label>)}
    {state?.window.click_through && <ActionButton kind="restore_click_through" variant="outline" className="w-full"><MousePointer2 />恢复点击</ActionButton>}
    {state?.window.always_on_top_status.detail && <p className="muted-note">{state.window.always_on_top_status.detail}</p>}
  </Panel></div></div>;
}
