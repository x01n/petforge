import type { ComponentProps, ReactNode } from "react";
import { LoaderCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { invoke, useControlSurface } from "@/lib/bridge";
import type { ActionPayload } from "@/lib/contract";

export function ActionButton({ kind, payload = {}, children, disabled, ...props }: ComponentProps<typeof Button> & { kind: string; payload?: ActionPayload }) {
  const key = kind + JSON.stringify(payload);
  const connected = useControlSurface((current) => current.connected);
  const waiting = useControlSurface((current) => current.pending.has(key));
  return <Button {...props} data-action={kind} disabled={!connected || disabled || waiting} onClick={() => { void invoke(kind, payload); }}>
    {waiting && <LoaderCircle className="animate-spin" />} {children}
  </Button>;
}

export function Choice({ label, value, options, kind, field, disabled, payload = {} }: {
  label: string; value?: string; options: { value: string; label: string; disabled?: boolean }[];
  kind: string; field: string; disabled?: boolean; payload?: ActionPayload;
}) {
  const connected = useControlSurface((current) => current.connected);
  const busy = useControlSurface((current) => [...current.pending].some((key) => key.startsWith(kind + "{")));
  return <Select value={value || undefined} disabled={!connected || disabled || busy || !options.length} onValueChange={(next) => { void invoke(kind, { ...payload, [field]: next }); }}>
    <SelectTrigger aria-label={label} data-action={kind} className="w-full"><SelectValue placeholder={options.length ? label : "暂无可用选项"} /></SelectTrigger>
    <SelectContent>{options.map((option) => <SelectItem key={option.value} value={option.value} disabled={option.disabled}>{option.label}</SelectItem>)}</SelectContent>
  </Select>;
}

export function Status({ good, children }: { good?: boolean; children: ReactNode }) {
  return <Badge variant="outline" className={good ? "status-ready" : "status-neutral"}><span className="status-dot" />{children || "等待状态"}</Badge>;
}

export function Panel({ title, description, action, children, className = "" }: { title: string; description?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return <Card className={className}><CardHeader className="panel-heading"><div className="min-w-0"><CardTitle>{title}</CardTitle>{description && <CardDescription>{description}</CardDescription>}</div>{action}</CardHeader><CardContent>{children}</CardContent></Card>;
}

export function EmptyState({ icon, title, description }: { icon: ReactNode; title: string; description: string }) {
  return <div className="empty-state"><div className="empty-icon">{icon}</div><h3>{title}</h3><p>{description}</p></div>;
}

export function Detail({ label, value }: { label: string; value: ReactNode }) {
  return <div className="detail-row"><span>{label}</span><strong>{value || "尚未上报"}</strong></div>;
}
