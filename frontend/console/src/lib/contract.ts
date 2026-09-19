import { z } from "zod";

const text = z.string().catch("").default("");
const flag = z.boolean().catch(false).default(false);
const count = z.number().finite().nonnegative().catch(0).default(0);
const strings = z.array(z.string()).catch([]).default([]);
const action = z.object({
  kind: z.string(), label: text, enabled: flag,
  payload: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])).catch({}).default({}),
});
const approval = z.object({ display_name: text, safe_summary: text, remaining_seconds: count, expires_at: count });
const operation = z.object({ label: text, status: text, status_label: text, message: text, updated_at: count });
const capability = z.object({ id: z.string(), label: text });
const diagnosticApi = z.object({
  kind: text, status: text, channel: text, model: text,
  first_ms: z.number().finite().nullable().catch(null).default(null),
  total_ms: z.number().finite().nullable().catch(null).default(null),
  tool_count: count, cache_read: flag, cache_write: flag, error_type: text,
});
const auditDuration = z.number().finite().nonnegative().max(1_000_000).nullable().catch(null).default(null);
const auditCount = z.number().int().nonnegative().max(1_000_000).catch(0).default(0);
const auditGroupRow = z.object({
  key: text, count: auditCount, avg_first_ms: auditDuration, avg_total_ms: auditDuration,
});
const auditSummary = z.object({
  total: auditCount,
  by_status: z.array(auditGroupRow).catch([]).default([]),
  by_channel: z.array(auditGroupRow).catch([]).default([]),
  latency_ms: z.object({
    avg_first: auditDuration, avg_total: auditDuration, max_total: auditDuration,
  }).prefault({}),
});
const diagnosticLog = z.object({
  level: text, logger: text, event: text, status: text,
  duration_ms: z.number().finite().nullable().catch(null).default(null),
  reason: text, detail: text,
});
const logSummaryCount = z.number().int().nonnegative().max(1_000_000).catch(0).default(0);
const logSummaryBins = z.record(z.string(), logSummaryCount).catch({}).default({});
// total 与 warning_plus 为可选：宿主当前投影只下发三个 by_* 维度，
// 缺少这两个字段的合法摘要不能因此整体回退为空。
// 坏结构走 nullable.catch(null)，与 api_audit.summary 的回退先例一致。
const logSummary = z.object({
  total: logSummaryCount.optional(),
  warning_plus: logSummaryCount.optional(),
  by_level: logSummaryBins,
  by_logger: logSummaryBins,
  by_event: logSummaryBins,
}).nullable().catch(null).default(null);

/** The public state is projected by gui.web.control_surface.public_control_state. */
export const controlStateSchema = z.object({
  revision: z.number().int().nonnegative(),
  updated_at: count,
  connection: z.object({ state: text, message: text }).prefault({}),
  interaction: z.object({
    phase: text, streaming: flag, busy: flag, text, murmur: text, tool_status: text,
    mood: text, safe_message: text, retryable: flag,
    actions: z.array(action).catch([]).default([]),
    approval: approval.nullable().catch(null).default(null),
    pending_approvals: z.array(approval).catch([]).default([]),
    model: z.object({ ready: z.boolean().nullable().catch(null).default(null), message: text, channel_count: count, action: z.array(action).catch([]).default([]) }).prefault({}),
  }).prefault({}),
  model_channels: z.array(z.object({
    id: z.string().min(1), protocol: text, model: text, models: strings,
    model_configured: flag, ready: flag, failures: count, active: flag, selectable: flag, status: text,
  })).catch([]).default([]),
  model_image: z.object({ ready: flag, mode: text, label: text }).prefault({}),
  renderer: z.object({ label: text, available: flag, message: text, model: text, models: strings, ready: flag, lifecycle_state: text, actual_api: text, frame_rate: count, geometry_audit_hz: count, rendered_frames: count, throttled_frames: count, geometry_audits: count, geometry_cache_hits: count, geometry_cache_misses: count, geometry_vertex_dirty_frames: count, geometry_structural_dirty_frames: count, frame_time_ms: z.number().finite().nonnegative().max(1_000_000).nullable().catch(null).default(null), large_frame_deltas: count }).prefault({}),
  tts: z.object({
    state: text, message: text, backend: text, available: flag, language: text, language_protocol: text, profile: text,
    profiles: z.array(z.object({ id: z.string().min(1), languages: strings, enabled: flag, active: flag })).catch([]).default([]),
    health: z.object({ checked: flag, available: flag, pending: flag, backend: text, message: text, latency_ms: z.number().finite().nullable().catch(null).default(null) }).prefault({}),
  }).prefault({}),
  window: z.object({
    locked: flag, click_through: flag, always_on_top: flag,
    position: z.object({ x: z.number(), y: z.number() }).nullable().catch(null).default(null),
    always_on_top_status: z.object({ status: text, enabled: flag, detail: text }).prefault({}),
    input_shape: z.object({ ready: flag, input_ready: flag, status: text }).prefault({}),
  }).prefault({}),
  affection: z.object({ current: text, tier: text, threshold: text, description: text, mood: text }).prefault({}),
  parts: z.array(z.object({ id: z.string(), label: text, description: text, enabled: flag })).catch([]).default([]),
  capabilities: z.object({ expressions: z.array(capability).catch([]).default([]), motions: z.array(capability).catch([]).default([]) }).prefault({}),
  operation: operation.prefault({}),
  timeline: z.array(operation).catch([]).default([]),
  observation: z.object({
    kind: text, label: text, status: text, status_label: text, message: text, title: text, process_name: text,
    pid: z.number().nullable().catch(null).default(null),
    processes: z.array(z.object({ name: text, pid: z.number().nullable().catch(null).default(null) })).catch([]).default([]),
  }).prefault({}),
  configuration: z.object({ connected: flag, watcher_enabled: flag, watcher_running: flag, auto_reload: flag, status: text, status_label: text, generation: count, message: text }).prefault({}),
  memory: z.object({
    enabled: flag, status: text, status_label: text, recall_limit: count, context_max_chars: count, max_memories: count,
    prune_importance_floor: count, exchange_importance: count, extract_default_priority: count,
    always_recall_priority: count, recall_min_similarity: z.number().finite().nonnegative().catch(0).default(0),
    auto_extract_enabled: flag,
    consolidation_enabled: flag, summarization_enabled: flag, summary_running: flag, summary_busy: flag,
    summary_completed: count, summary_last_status: text, extraction_pending: count, extraction_completed: count,
    extraction_last_status: text, vector_index_size: count, lexical_index: text, message: text,
    recall_total: count, recall_hit_rate: z.number().finite().nonnegative().catch(0).default(0),
    recall_top_mean_score: z.number().finite().nonnegative().catch(0).default(0),
    recall_calibration: z.number().finite().nonnegative().catch(0).default(0),
  }).prefault({}),
  activity: z.object({ system_idle_provider: text, system_idle_provider_label: text, system_idle_status: text, system_idle_status_label: text, system_idle_available: flag }).prefault({}),
  scheduler: z.object({
    status: text, running: flag, task_count: count, trigger_count: count, event_count: count,
    matched_count: count, completed_count: count, failed_count: count, skipped_count: count,
    last_status: text, last_duration_ms: z.number().finite().nullable().catch(null).default(null),
  }).prefault({}),
  proactive: z.object({
    status: text, enabled: flag, running: flag, rule_count: count, pending_events: count,
    waiting_for_conversation: flag, hourly_used: count, hourly_budget: count,
    daily_used: count, daily_budget: count, accepted: count, completed: count,
    dropped: count, budget_rejections: count,
  }).prefault({}),
  diagnostics: z.object({
    api_audit: z.object({ status: text, count, summary: auditSummary.nullable().catch(null).default(null), records: z.array(diagnosticApi).catch([]).default([]) }).prefault({}),
    logs: z.object({ status: text, count, summary: logSummary, records: z.array(diagnosticLog).catch([]).default([]) }).prefault({}),
  }).prefault({}),
  theme: z.record(z.string(), z.string()).catch({}).default({}),
});

export type ControlState = z.infer<typeof controlStateSchema>;
export type Action = z.infer<typeof action>;
export type ActionPayload = Record<string, unknown>;
export type ActionResult = { status: string; status_label?: string; message?: string; reason?: string; detail?: string };
export type Bridge = {
  getState(callback: (json: string) => void): void;
  invoke(kind: string, payload: string, callback: (json: string) => void): void;
};

export const actionResultSchema = z.object({ status: z.string(), status_label: text, message: text, reason: text, detail: text });

export function parseState(raw: unknown): ControlState | null {
  try {
    const result = controlStateSchema.safeParse(typeof raw === "string" ? JSON.parse(raw) : raw);
    return result.success ? result.data : null;
  } catch { return null; }
}

export function parseResult(raw: unknown): ActionResult {
  try {
    const result = actionResultSchema.safeParse(typeof raw === "string" ? JSON.parse(raw) : raw);
    if (result.success) return result.data;
  } catch { /* Invalid bridge replies are surfaced without displaying raw data. */ }
  return { status: "unavailable", message: "操作回执格式无效" };
}
