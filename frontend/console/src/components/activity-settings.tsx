import { useMemo, useState } from "react";
import { Activity as ActivityIcon, Brain, Clock3, FileSliders, Focus, ListTree, Monitor, RefreshCw, Search, Settings2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ActionButton, Detail, EmptyState, Panel, Status } from "./console-controls";
import type { ControlState } from "@/lib/contract";

export function Timeline({ state }: { state: ControlState | null }) {
  const timeline = state?.timeline ?? [];
  return <Panel title="最近活动" description="桌宠操作与执行回执" action={<Clock3 className="section-icon" />}>
    {timeline.length ? <ScrollArea className="timeline-scroll"><ol className="timeline-list">{[...timeline].reverse().map((item, index) => <li key={`${item.updated_at}-${index}`}><span className={`timeline-dot ${["failed", "error", "timeout", "unavailable"].includes(item.status) ? "timeline-error" : ""}`} /><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center justify-between gap-2"><strong>{item.label || "桌宠操作"}</strong><time>{item.updated_at ? new Date(item.updated_at * 1000).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }) : ""}</time></div><p>{item.message || item.status_label}</p><span className="timeline-status">{item.status_label}</span></div></li>)}</ol></ScrollArea> : <EmptyState icon={<Clock3 />} title="暂时没有活动" description="发送消息或操作桌宠后，执行状态会显示在这里。" />}
  </Panel>;
}

export function ActivitySettings({ state }: { state: ControlState | null }) {
  const memory = state?.memory;
  const observation = state?.observation;
  const diagnostics = state?.diagnostics;
  const [logLevel, setLogLevel] = useState("ALL");
  const [logQuery, setLogQuery] = useState("");
  const filteredLogs = useMemo(() => {
    const query = logQuery.trim().toLowerCase();
    return (diagnostics?.logs.records ?? []).filter((row) => {
      const matchesLevel = logLevel === "ALL" || row.level === logLevel;
      const searchable = [row.event, row.logger, row.status, row.reason, row.detail]
        .join(" ")
        .toLowerCase();
      return matchesLevel && (!query || searchable.includes(query));
    });
  }, [diagnostics?.logs.records, logLevel, logQuery]);
  const logSummary = diagnostics?.logs.summary;
  // 分组键由宿主 _safe_log_diagnostic 白名单投影，按计数降序投递；
  // 这里只取前 5 项，与 API 审计摘要的“键 x 计数”文案保持一致。
  const logSummaryBins = (bins: Record<string, number>) => Object.entries(bins).slice(0, 5).map(([key, value]) => `${key} x ${value}`).join(" ｜ ");
  // 聚合分组键由宿主按 "<prefix>:" 构造（PREFIX:value）；前端不做固定长度
  // 前缀裁剪："status:"/"channel:" 前缀精确匹配时去除，其余键原样展示。
  const aggregateLabel = (raw: string, prefix: string) => (raw.startsWith(prefix) ? raw.slice(prefix.length) : raw);
  const logSummaryVisible = Boolean(logSummary && (logSummary.total != null || logSummary.warning_plus != null || Object.keys(logSummary.by_level).length || Object.keys(logSummary.by_logger).length || Object.keys(logSummary.by_event).length));
  return <div className="settings-grid"><div className="space-y-5"><Panel title="记忆状态" description="持续积累，让下一次对话更有默契" action={<Status good={memory?.enabled}>{memory?.status_label || "等待记忆状态"}</Status>}>
    <div className="memory-metrics"><div><Brain /><strong>{state ? memory?.vector_index_size.toLocaleString() : "—"}</strong><span>向量索引</span></div><div><ListTree /><strong>{state ? memory?.recall_limit : "—"}</strong><span>每次召回上限</span></div></div>
    <div className="details"><Detail label="召回命中率" value={state ? `${(memory?.recall_hit_rate ?? 0) * 100}% / ${memory?.recall_total ?? 0} 次` : "尚未上报"} /><Detail label="记忆容量上限" value={state ? memory?.max_memories.toLocaleString() : "尚未上报"} /><Detail label="上下文字符上限" value={state ? memory?.context_max_chars.toLocaleString() : "尚未上报"} /><Detail label="记忆整合" value={memory?.consolidation_enabled ? "已启用" : "未启用"} /><Detail label="自动摘要" value={memory?.summarization_enabled ? memory.summary_busy ? "正在整理" : "已启用" : "未启用"} /><Detail label="自动提取" value={memory?.auto_extract_enabled ? "已启用" : "未启用"} /><Detail label="优先级 / 必召回阈值" value={state ? `${memory?.extract_default_priority} / ${memory?.always_recall_priority}` : "尚未上报"} /><Detail label="交换记忆优先级" value={state ? String(memory?.exchange_importance) : "尚未上报"} /><Detail label="淘汰优先级下限" value={state ? String(memory?.prune_importance_floor) : "尚未上报"} /><Detail label="相似度下限" value={state ? memory?.recall_min_similarity.toFixed(2) : "尚未上报"} /><Detail label="已完成摘要" value={state ? String(memory?.summary_completed) : "尚未上报"} /><Detail label="待提取 / 已提取" value={state ? `${memory?.extraction_pending} / ${memory?.extraction_completed}` : "尚未上报"} /></div>
    <p className="muted-note">{memory?.message || "连接后查看记忆系统运行状态。"}</p><ActionButton kind="open_config" payload={{ section: "memory" }} variant="outline" className="w-full"><Settings2 />管理记忆策略</ActionButton>
  </Panel><Panel title="桌面感知" description="主动读取当前窗口或运行中的程序" action={<Monitor className="section-icon" />}>
    <div className="flex flex-wrap gap-2"><ActionButton kind="read_foreground_window" variant="outline"><Focus />读取当前窗口</ActionButton><ActionButton kind="read_processes" variant="outline"><ListTree />读取运行程序</ActionButton></div>
    <div className="details"><Detail label="活动感知" value={state?.activity.system_idle_provider_label} /><Detail label="探针状态" value={state?.activity.system_idle_status_label} /></div>
    {observation?.kind === "foreground_window" && <div className="observation"><strong>{observation.title || "未获取到窗口标题"}</strong><p>{observation.process_name}{observation.pid != null ? ` · PID ${observation.pid}` : ""}</p><Badge variant="outline">{observation.status_label}</Badge></div>}
    {observation?.kind === "processes" && <ScrollArea className="process-scroll"><div className="details">{observation.processes.map((process, index) => <Detail key={`${process.pid}-${index}`} label={process.name} value={process.pid == null ? "未知 PID" : String(process.pid)} />)}</div></ScrollArea>}
    {observation?.message && <p className="muted-note">{observation.message}</p>}
  </Panel><Panel title="主动行为与调度" description="查看定时任务、窗口触发和预算状态" action={<ActivityIcon className="section-icon" />}>
    <div className="details"><Detail label="主动行为" value={state?.proactive.enabled ? state.proactive.running ? "运行中" : "已启用" : "未启用"} /><Detail label="规则 / 待处理事件" value={state ? `${state.proactive.rule_count} / ${state.proactive.pending_events}` : "尚未上报"} /><Detail label="小时预算" value={state ? `${state.proactive.hourly_used} / ${state.proactive.hourly_budget}` : "尚未上报"} /><Detail label="每日预算" value={state ? `${state.proactive.daily_used} / ${state.proactive.daily_budget}` : "尚未上报"} /><Detail label="定时任务 / 触发器" value={state ? `${state.scheduler.task_count} / ${state.scheduler.trigger_count}` : "尚未上报"} /><Detail label="最近触发状态" value={state?.scheduler.last_status || "尚未上报"} /></div>
    <p className="muted-note">预算拒绝 {state?.proactive.budget_rejections ?? 0} 次，丢弃 {state?.proactive.dropped ?? 0} 次。</p><div className="flex flex-wrap gap-2"><ActionButton kind="open_config" payload={{ section: "proactive" }} variant="outline"><Settings2 />配置主动规则</ActionButton><ActionButton kind="open_config" payload={{ section: "scheduler" }} variant="outline"><Clock3 />配置定时触发</ActionButton></div>
  </Panel><Panel title="诊断与日志" description="查看模型调用、工具执行与运行时日志" action={<ActivityIcon className="section-icon" />}>
    <div className="flex flex-wrap gap-2"><ActionButton kind="read_api_audit" variant="outline"><ListTree />读取 API 调用审计</ActionButton><ActionButton kind="read_log_records" variant="outline"><Clock3 />读取运行日志</ActionButton></div>
    <p className="muted-note">结果会通过本地控制通道异步返回；敏感字段由宿主统一脱敏。</p>
    <div className="diagnostic-list"><strong>API 调用 {diagnostics?.api_audit.count ?? 0} 条</strong>
      {diagnostics?.api_audit.summary ? <>
        <div className="diagnostic-list"><strong>聚合摘要 · 总计 {diagnostics.api_audit.summary.total.toLocaleString()}</strong>
          {diagnostics.api_audit.summary.by_status.length ? <div className="diagnostic-row"><span>状态</span><small>{diagnostics.api_audit.summary.by_status.map((row) => `${aggregateLabel(row.key, "status:")} x ${row.count}`).join(" ｜ ")}</small></div> : null}
          {diagnostics.api_audit.summary.by_channel.length ? <div className="diagnostic-row"><span>渠道</span><small>{diagnostics.api_audit.summary.by_channel.map((row) => `${aggregateLabel(row.key, "channel:")} x ${row.count}`).join(" ｜ ")}</small></div> : null}
          <div className="diagnostic-row"><span>耗时</span><small>
            {diagnostics.api_audit.summary.latency_ms.avg_first != null ? `平均首字 ${Math.round(diagnostics.api_audit.summary.latency_ms.avg_first)}ms` : "暂无首字样本"}
            {diagnostics.api_audit.summary.latency_ms.avg_total != null ? ` · 平均总计 ${Math.round(diagnostics.api_audit.summary.latency_ms.avg_total)}ms` : ""}
            {diagnostics.api_audit.summary.latency_ms.max_total != null ? ` · 最长 ${Math.round(diagnostics.api_audit.summary.latency_ms.max_total)}ms` : ""}
          </small></div>
        </div>
      </> : null}
      <strong>最近记录</strong>{diagnostics?.api_audit.records.slice(0, 4).map((row, index) => <div className="diagnostic-row" key={`${row.channel}-${index}`}><span>{row.model || row.kind || "调用"}</span><small>{row.first_ms != null ? `首字 ${Math.round(row.first_ms)}ms` : "暂无首字"}{row.total_ms != null ? ` · 总计 ${Math.round(row.total_ms)}ms` : ""}{row.tool_count ? ` · 工具 ${row.tool_count}` : ""}</small></div>)}
    </div>
    <div className="log-toolbar"><div className="log-search"><Search /><Input aria-label="搜索运行日志" placeholder="搜索事件、组件或状态" value={logQuery} onChange={(event) => setLogQuery(event.target.value)} /></div><Select value={logLevel} onValueChange={setLogLevel}><SelectTrigger aria-label="日志等级"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="ALL">全部等级</SelectItem><SelectItem value="DEBUG">DEBUG</SelectItem><SelectItem value="INFO">INFO</SelectItem><SelectItem value="WARNING">WARNING</SelectItem><SelectItem value="ERROR">ERROR</SelectItem><SelectItem value="CRITICAL">CRITICAL</SelectItem></SelectContent></Select></div>
    <div className="diagnostic-list"><strong>运行日志 {diagnostics?.logs.count ?? 0} 条 · 当前显示 {filteredLogs.length} 条</strong>{logSummary && logSummaryVisible ? <><strong>日志聚合摘要{logSummary.total != null ? ` · 总计 ${logSummary.total.toLocaleString()}` : ""}{logSummary.warning_plus != null ? ` · 告警及以上 ${logSummary.warning_plus.toLocaleString()} 条` : ""}</strong>{Object.keys(logSummary.by_level).length ? <div className="diagnostic-row"><span>级别 Top5</span><small>{logSummaryBins(logSummary.by_level)}</small></div> : null}{Object.keys(logSummary.by_logger).length ? <div className="diagnostic-row"><span>来源 Top5</span><small>{logSummaryBins(logSummary.by_logger)}</small></div> : null}{Object.keys(logSummary.by_event).length ? <div className="diagnostic-row"><span>事件 Top5</span><small>{logSummaryBins(logSummary.by_event)}</small></div> : null}</> : null}<ScrollArea className="log-stream">{filteredLogs.length ? filteredLogs.map((row, index) => <article className={`log-row log-level-${row.level.toLowerCase()}`} key={`${row.event}-${row.logger}-${index}`}><div className="log-row-head"><Badge variant="outline">{row.level || "INFO"}</Badge><strong>{row.event || row.logger || "运行事件"}</strong><small>{row.logger || "app"}{row.status ? ` · ${row.status}` : ""}</small></div><p>{row.detail || row.reason || "已记录"}</p></article>) : <p className="muted-note">没有符合筛选条件的日志。</p>}</ScrollArea></div>
  </Panel></div><Timeline state={state} /></div>;
}

export function Configuration({ state }: { state: ControlState | null }) {
  const config = state?.configuration;
  return <div className="settings-width"><Panel title="配置与热重载" description="连接文件配置，修改后自动校验并应用" action={<FileSliders className="section-icon" />}>
    <div className="configuration-status"><div className="config-symbol"><RefreshCw /></div><div><Status good={config?.connected && config?.auto_reload}>{config?.status_label || "等待配置状态"}</Status><p>{config?.message || "尚未连接桌宠配置服务。"}</p></div></div>
    <div className="details"><Detail label="配置文件" value={config?.connected ? "已连接" : "未连接"} /><Detail label="自动重载" value={config?.auto_reload ? "已启用" : "未启用"} /><Detail label="文件监视器" value={config?.watcher_running ? "运行中" : "未运行"} /><Detail label="当前配置代次" value={state ? String(config?.generation) : "尚未上报"} /></div>
    <div className="flex flex-wrap gap-2"><ActionButton kind="open_config"><Settings2 />打开配置中心</ActionButton><ActionButton kind="restart_application" variant="outline"><RefreshCw />重启桌宠</ActionButton></div>
    <p className="muted-note">模型连接、模块配置和高级选项在桌面配置中心管理。</p>
  </Panel></div>;
}
