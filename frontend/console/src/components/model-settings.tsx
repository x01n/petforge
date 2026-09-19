import { AudioLines, BrainCircuit, Image, Radio, Settings2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ActionButton, Choice, Detail, EmptyState, Panel, Status } from "./console-controls";
import type { ControlState } from "@/lib/contract";

const languageLabels: Record<string, string> = { zh: "中文", en: "英语", ja: "日语", ko: "韩语", yue: "粤语", auto: "自动识别", all_zh: "中文", all_ja: "日语", all_ko: "韩语", all_yue: "粤语" };

export function VoiceSettings({ state }: { state: ControlState | null }) {
  const tts = state?.tts;
  const active = tts?.profiles.find((profile) => profile.id === tts.profile || profile.active);
  const languages = [...new Set([...(active?.languages ?? []), ...(tts?.language ? [tts.language] : [])])];
  return <Panel title="声音与语言" description="为每一段对话选择合适的声音" action={<AudioLines className="section-icon" />}>
    <div className="field-stack"><label>当前语音模型</label><Choice label="选择语音模型" value={tts?.profile} kind="select_tts_profile" field="profile" options={tts?.profiles.map((profile) => ({ value: profile.id, label: profile.id, disabled: !profile.enabled })) ?? []} /></div>
    <div className="field-stack"><label>输出语言</label><Choice label="选择输出语言" value={tts?.language} kind="select_tts_language" field="language" options={languages.map((language) => ({ value: language, label: languageLabels[language] ?? language }))} /></div>
    <div className="details"><Detail label="合成状态" value={tts?.state} /><Detail label="语音后端" value={tts?.backend} /><Detail label="健康检查" value={tts?.health.checked ? <Status good={tts.health.available}>{tts.health.pending ? "初始化中" : tts.health.available ? "已就绪" : "未就绪"}</Status> : "尚未检测"} />{tts?.health.latency_ms != null && <Detail label="检测耗时" value={`${Math.round(tts.health.latency_ms)} ms`} />}</div>
    <p className="muted-note">{tts?.message || tts?.health.message || "连接桌宠后显示实际语音配置与运行状态。"}</p>
    <ActionButton kind="open_config" payload={{ section: "tts" }} variant="outline" className="w-full"><Settings2 />管理语音配置</ActionButton>
  </Panel>;
}

export function ModelSettings({ state }: { state: ControlState | null }) {
  return <Tabs defaultValue="models" className="space-y-5"><TabsList aria-label="模型与语音分类"><TabsTrigger value="models"><BrainCircuit />模型渠道</TabsTrigger><TabsTrigger value="voice"><AudioLines />语音输出</TabsTrigger></TabsList>
    <TabsContent value="models"><div className="models-layout"><Panel title="模型渠道" description="实际连接状态、模型选择与错误重试信息" action={<ActionButton kind="configure_model" payload={{ target: "model_setup", mode: "click" }} variant="outline" size="sm"><Settings2 />配置模型</ActionButton>}>
      {state?.model_channels.length ? <div className="channel-list">{state.model_channels.map((channel) => <div className={`channel-card ${channel.active ? "channel-active" : ""}`} key={channel.id}>
        <div className="channel-heading"><div className="channel-logo"><Radio /></div><div className="min-w-0 flex-1"><h3>{channel.id}</h3><p>{channel.protocol}</p></div>{channel.active && <Badge>使用中</Badge>}</div>
        <div className="flex flex-wrap gap-2"><Status good={channel.ready}>{channel.status}</Status>{channel.failures > 0 && <Badge variant="outline">连续失败 {channel.failures} 次</Badge>}</div>
        <Choice label={`${channel.id} 的模型`} value={channel.model} options={[...new Set([...channel.models, ...(channel.model ? [channel.model] : [])])].map((model) => ({ value: model, label: model }))} kind="select_model_channel" payload={{ channel_id: channel.id }} field="model" disabled={!channel.selectable} />
        {!channel.active && <ActionButton kind="select_model_channel" payload={{ channel_id: channel.id }} variant="outline" size="sm" disabled={!channel.selectable}>切换到此渠道</ActionButton>}
      </div>)}</div> : <EmptyState icon={<BrainCircuit />} title="还没有模型渠道" description={state?.interaction.model.message || "连接桌宠后，可在配置模型中添加你的模型服务。"} />}
    </Panel><div className="space-y-5"><Panel title="图片理解" description="识图能力由当前模型配置决定" action={<Image className="section-icon" />}><Status good={state?.model_image.ready}>{state?.model_image.label || "等待模型状态"}</Status><p className="muted-note">主模型与视觉模型的连接方式可以在配置中心管理。</p><ActionButton kind="open_config" payload={{ section: "llm" }} variant="outline" className="w-full">打开模型配置</ActionButton></Panel><VoiceSettings state={state} /></div></div></TabsContent>
    <TabsContent value="voice"><div className="settings-width"><VoiceSettings state={state} /></div></TabsContent>
  </Tabs>;
}
