import { memo, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { ArrowUp, Check, CornerDownLeft, MessageCircle, RotateCcw, ShieldCheck, Square, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ActionButton, EmptyState, Panel } from "./console-controls";
import { invoke, useControlSurface } from "@/lib/bridge";
import type { ControlState } from "@/lib/contract";

type Approval = ControlState["interaction"]["pending_approvals"][number];

function approvalsEqual(previous: ReadonlyArray<Approval>, next: ReadonlyArray<Approval>) {
  if (previous === next) return true;
  if (previous.length !== next.length) return false;
  return previous.every((item, index) => {
    const other = next[index];
    return item.display_name === other.display_name
      && item.safe_summary === other.safe_summary
      && item.remaining_seconds === other.remaining_seconds
      && item.expires_at === other.expires_at;
  });
}

const ApprovalBox = memo(function ApprovalBox({ approvals }: { approvals: ReadonlyArray<Approval> }) {
  const [clock, setClock] = useState(Date.now());
  useEffect(() => {
    if (!approvals.length) return;
    const timer = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [approvals.length]);
  return <div className="approval-box" role="region" aria-label="待确认操作">
    <div className="approval-heading"><ShieldCheck /><strong>需要你的确认</strong><Badge variant="outline">{approvals.length} 项操作</Badge></div>
    <ApprovalList approvals={approvals} clock={clock} />
    <div className="flex flex-wrap gap-2">
      <ActionButton kind="approve" size="sm"><Check />允许一次</ActionButton>
      <ActionButton kind="grant_session" size="sm" variant="outline">本次会话允许</ActionButton>
      <ActionButton kind="deny" size="sm" variant="ghost"><X />拒绝</ActionButton>
    </div>
  </div>;
}, (previous, next) => approvalsEqual(previous.approvals, next.approvals));

/** 逐项渲染审批卡片：key 由 display_name + expires_at 构成，杜绝秒级
    时钟变化导致重复元素复用错位；时钟刷新只触发本列表重渲染。 */
function ApprovalList({ approvals, clock }: { approvals: ReadonlyArray<Approval>; clock: number }) {
  return <>
    {approvals.map((approval) => {
      const localRemaining = Math.max(0, Math.ceil(approval.expires_at - clock / 1000));
      const remaining = Math.min(localRemaining, approval.remaining_seconds);
      return <div key={`${approval.display_name}-${approval.expires_at}`}><strong>{approval.display_name}</strong><p>{approval.safe_summary}</p><small>{remaining} 秒后过期</small></div>;
    })}
  </>;
}

export function Conversation({ state, compact = false }: { state: ControlState | null; compact?: boolean }) {
  const connected = useControlSurface((current) => current.connected);
  const sending = useControlSurface((current) => [...current.pending].some((key) => key.startsWith("submit_text{")));
  const [draft, setDraft] = useState("");
  const [sent, setSent] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  // 同一消息的重复提交防护：事件处理期间同步置位并在回执后复位。
  const submitGuard = useRef(false);
  // 保证 React 从稳定的 ref 读取最新处理器，避免闭包读到过期的
  // sending / connected / interaction 快照（StrictMode 双调用下有防线）。
  const handleSubmitRef = useRef<(event: FormEvent) => void>(() => {});
  const interaction = state?.interaction;
  const approvals = useMemo(
    () => interaction?.pending_approvals.length ? interaction.pending_approvals : interaction?.approval ? [interaction.approval] : [],
    [interaction?.pending_approvals, interaction?.approval],
  );
  useEffect(() => {
    if (follow.current) bottom.current?.scrollIntoView({ block: "nearest" });
  }, [interaction?.text, interaction?.murmur, interaction?.tool_status, sent]);
  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || !connected || interaction?.busy || sending || submitGuard.current) return;
    // invoke 是同步建立 pending 键的，置位 ref 后同一事件循环内的重复
    // submit 会被拒；回执（成功/失败）后允许再次发送同一条文字。
    submitGuard.current = true;
    const result = await invoke("submit_text", { text });
    submitGuard.current = false;
    if (!["failed", "error", "unavailable", "denied", "timeout"].includes(result.status)) {
      setSent(text);
      setDraft("");
      follow.current = true;
    }
  }
  handleSubmitRef.current = submit;
    return <Panel title="和桌宠聊聊" description={connected ? interaction?.phase || "等待互动" : "从桌宠连接后开始对话"} className={compact ? "conversation-panel compact" : "conversation-panel"}
    action={<Badge variant="secondary">{interaction?.streaming ? "流式回复中" : "对话"}</Badge>}>
    <div className="conversation-scroll" ref={scroll} onScroll={() => { if (scroll.current) follow.current = scroll.current.scrollHeight - scroll.current.scrollTop - scroll.current.clientHeight < 80; }}>
      {sent && <div className="message user-message"><span className="message-author">你 · 本次发送</span><p>{sent}</p></div>}
      {interaction?.text ? <div className="message pet-message"><span className="message-author"><span className="pet-dot" /> MeaPet {interaction.streaming && <span className="stream-dot" />}</span><p>{interaction.text}</p></div>
        : <EmptyState icon={<MessageCircle />} title={connected ? "今天，有什么想分享？" : "等待与你的桌宠连接"} description={connected ? "说说今天的事，或者让桌宠帮你完成一个小任务。" : "请通过桌宠托盘打开控制台。连接后，对话与语音状态会实时显示在这里。"} />}
      {interaction?.murmur && <details className="murmur"><summary>桌宠的碎碎念</summary><p>{interaction.murmur}</p></details>}
      {interaction?.tool_status && <div className="inline-notice"><ShieldCheck />{interaction.tool_status}</div>}
      {interaction?.safe_message && <p className="muted-note">{interaction.safe_message}</p>}
      <div ref={bottom} />
    </div>
    {approvals.length > 0 && <ApprovalBox approvals={approvals} />}
    <form className="composer" onSubmit={handleSubmitRef.current}>
      <Textarea aria-label="发送给桌宠的消息" placeholder="输入消息，和桌宠聊聊…" maxLength={2000} value={draft} disabled={!connected} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => {
        if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); }
      }} />
      <div className="composer-footer"><span className="composer-hint"><CornerDownLeft />发送 · Shift + Enter 换行</span><div className="flex gap-2">
        {interaction?.busy && <ActionButton kind="stop" size="sm" variant="outline"><Square />停止</ActionButton>}
        {interaction?.retryable && <ActionButton kind="retry" size="sm" variant="outline"><RotateCcw />重试</ActionButton>}
        <Button data-action="submit_text" type="submit" size="icon" aria-label="发送消息" disabled={!connected || !draft.trim() || interaction?.busy || sending}><ArrowUp /></Button>
      </div></div>
    </form>
  </Panel>;
}
