<script setup>
import { computed } from 'vue'
import { briefCitations, briefTextParts } from '../briefs.mjs'
import { mineTime } from '../utils.mjs'
import { assistantChannel } from '../assistant.mjs'

const props = defineProps({ brief: Object, loading: Boolean, error: String, actionBusy: Boolean, pollPaused: Boolean,
  citationDetail: Object, citationLoading: Boolean, citationError: String })
const emit = defineEmits(['generate', 'retry', 'cancel', 'refresh', 'open-citation', 'close-citation'])
const task = computed(() => props.brief?.task)
const citations = computed(() => briefCitations(props.brief))
const parts = computed(() => briefTextParts(task.value?.final_text || '', citations.value))
const states = { not_started: '尚未生成', queued: '排队中', running: '执行中', completed: '已完成', partial: '部分完成', failed: '生成失败', cancelled: '已取消' }
const tools = { query_series: '查询监测序列', run_prediction: '运行预测', get_forecast: '读取预测结果', query_events: '查询预警事件', get_event_evidence: '读取事件依据', search_documents: '检索批准资料', draft_advice: '生成建议草稿', get_case: '查询处置进展' }
const stepStates = { queued: '排队中', started: '执行中', running: '执行中', success: '已完成', completed: '已完成', error: '失败', failed: '失败', cancelled: '已取消' }
const kinds = { event: '预警事件', rule: '预警规则', observation: '监测记录', forecast_run: '预测结果', reference: '资料原文', attribution_run: '溯源结果' }
const ruleFields = { rule_version: '规则版本', status: '规则状态', observed_red_threshold: '实测红色阈值', observed_orange_threshold: '实测橙色阈值', orange_duration_seconds: '橙色持续时间 / 秒', forecast_yellow_threshold: '预测黄色阈值', forecast_orange_threshold: '预测橙色阈值' }
function argumentText(argumentsValue) {
  if (!argumentsValue || typeof argumentsValue !== 'object') return ''
  const labels = { event_id: '事件', event_revision: '修订', point_id: '测点', run_id: '预测运行', query: '关键词', start_at: '开始', end_at: '结束' }
  return Object.entries(argumentsValue).filter(([key]) => labels[key]).map(([key, value]) => `${labels[key]}：${key.endsWith('_at') ? mineTime(value) : String(value).slice(0, 140)}`).join(' · ')
}
const detailIntervals = computed(() => {
  const outputs = props.citationDetail?.forecast?.outputs || []
  return (outputs.find(item => item.prediction_stage === 'residual_corrected' && item.quantity_semantics === 'physical_forecast') || outputs.find(item => item.quantity_semantics === 'physical_forecast'))?.interval_values || []
})
</script>

<template>
<section class="event-brief" aria-label="研判简报" aria-live="polite">
  <div class="brief-heading"><h3>研判简报</h3><span :class="['brief-state', brief?.status]">{{ loading ? '读取中' : states[brief?.status || 'not_started'] || '待确认' }}</span></div>
  <p v-if="brief?.stale" class="brief-version">基于第{{ brief.event_revision }}版 · 当前第{{ brief.requested_event_revision }}版</p>
  <p v-else-if="brief?.event_revision" class="brief-version">事件第 {{ brief.event_revision }} 版</p>
  <p v-if="brief?.as_of" class="brief-meta">数据截止 {{ mineTime(brief.as_of) }}</p>
  <p v-if="task" class="brief-meta">{{ assistantChannel(task) === 'DeepSeek' ? task.model : assistantChannel(task) === '本机查询' ? '本机依据' : '业务查询' }}<span v-if="brief.attempt"> · 第 {{ brief.attempt }} 次任务</span></p>
  <p v-if="error" class="brief-message warning">{{ error }}</p>
  <p v-if="pollPaused" class="brief-message">任务仍在后台执行，可刷新查看进度。</p>
  <div v-if="parts.length" class="brief-text"><template v-for="(part,index) in parts" :key="index"><button v-if="part.citation" class="inline-citation" :aria-label="`正文依据 ${part.citation.citation_id}`" @click="emit('open-citation', part.citation)">{{ part.text }}</button><span v-else>{{ part.text }}</span></template></div>
  <p v-else-if="brief?.status === 'queued' || brief?.status === 'running'" class="brief-message">正在读取事件依据并生成研判简报…</p>
  <p v-else-if="!loading && (!brief || brief.status === 'not_started')" class="brief-message">该事件尚无研判简报，可生成当前版本。</p>
  <div class="brief-actions">
    <button v-if="!brief || brief.status === 'not_started' || brief.stale" class="brief-button primary" :disabled="loading || actionBusy" @click="emit('generate')">{{ brief?.stale ? '生成当前版本' : '生成简报' }}</button>
    <button v-if="brief?.can_retry && !brief?.stale" class="brief-button primary" :disabled="actionBusy" @click="emit('retry')">重试简报</button>
    <button v-if="brief?.can_cancel" class="brief-button" :disabled="actionBusy" @click="emit('cancel')">取消简报</button>
    <button class="brief-button" :disabled="loading || actionBusy" @click="emit('refresh')">刷新简报</button>
  </div>

  <section v-if="task?.tool_steps?.length || task?.tool_calls?.length" class="brief-section" aria-label="工具调用过程">
    <h3>工具调用过程</h3>
    <ol class="brief-steps"><li v-for="(step,index) in task.tool_steps || task.tool_calls" :key="step.call_id || index"><div><strong>{{ index + 1 }}. {{ tools[step.name] || '业务查询' }}</strong><span :class="['step-state',step.status]">{{ stepStates[step.status] || '待确认' }}</span></div><p v-if="argumentText(step.arguments)">{{ argumentText(step.arguments) }}</p><small v-if="Number.isFinite(step.elapsed_ms) && !['running','started'].includes(step.status)">耗时 {{ step.elapsed_ms }} ms</small><p v-if="['error','failed'].includes(step.status)" class="warning">该步骤未完成，已有结果保留。</p></li></ol>
  </section>
  <section v-if="citations.length" class="brief-section" aria-label="依据引用">
    <h3>依据引用</h3><ul class="brief-citations"><li v-for="citation in citations" :key="citation.citation_id"><button :aria-label="`打开依据 ${citation.citation_id}`" @click="emit('open-citation', citation)"><b>[{{ citation.citation_id }}]</b><span>{{ citation.label || kinds[citation.kind] }}<small>{{ kinds[citation.kind] }}{{ citation.locator?.page ? ` · 第 ${citation.locator.page} 页` : '' }}</small></span><i>↗</i></button></li></ul>
  </section>
  <section v-if="citationDetail || citationLoading || citationError" class="brief-section citation-detail" aria-label="引用依据">
    <div class="brief-heading"><h3>{{ citationDetail?.title || '引用依据' }}</h3><button class="brief-button" @click="emit('close-citation')">关闭依据</button></div>
    <p v-if="citationLoading" class="brief-message">正在读取原始依据…</p><p v-if="citationError" class="brief-message warning">{{ citationError }}</p>
    <template v-if="citationDetail">
      <p v-if="citationDetail.summary" class="brief-meta">{{ citationDetail.summary }}</p>
      <dl v-if="citationDetail.event" class="brief-facts"><div><dt>事件</dt><dd>{{ citationDetail.event.event_id }}</dd></div><div><dt>修订</dt><dd>第 {{ citationDetail.event.event_revision }} 版</dd></div><div><dt>测点</dt><dd>{{ citationDetail.event.point_id }}</dd></div><div><dt>开始时间</dt><dd>{{ mineTime(citationDetail.event.start_at) }}</dd></div></dl>
      <dl v-if="citationDetail.rules" class="brief-facts"><template v-for="(label,key) in ruleFields" :key="key"><div v-if="citationDetail.rules[key] !== undefined"><dt>{{ label }}</dt><dd>{{ citationDetail.rules[key] === 'team-draft' ? '团队草案' : citationDetail.rules[key] }}</dd></div></template></dl>
      <div v-if="citationDetail.observations?.length" class="brief-table"><table><thead><tr><th>观测时间</th><th>原始值</th><th>有效值</th><th>原始行</th></tr></thead><tbody><tr v-for="row in citationDetail.observations" :key="row.id || row.observation_id"><td>{{ mineTime(row.event_time) }}</td><td>{{ row.raw_value ?? '—' }}</td><td>{{ row.value ?? '缺测' }}</td><td>{{ row.row_number ?? '—' }}</td></tr></tbody></table></div>
      <template v-if="citationDetail.forecast"><p class="brief-meta">{{ citationDetail.forecast.model_id }} · {{ citationDetail.forecast.model_version }}</p><p class="brief-meta">输入截止 {{ mineTime(citationDetail.forecast.cutoff) }}</p><div class="brief-table"><table><thead><tr><th>区间开始</th><th>区间结束</th><th>预测值</th></tr></thead><tbody><tr v-for="interval in detailIntervals" :key="interval.start_at"><td>{{ mineTime(interval.start_at) }}</td><td>{{ mineTime(interval.end_at) }}</td><td>{{ Number(interval.value).toFixed(3) }}</td></tr></tbody></table></div></template>
      <article v-for="fragment in citationDetail.fragments || []" :key="fragment.reference_id" class="brief-fragment"><p>{{ fragment.text }}</p><small>{{ fragment.source }}{{ fragment.locator?.page ? ` · 第 ${fragment.locator.page} 页` : '' }}</small></article>
      <template v-if="citationDetail.attribution"><p>{{ citationDetail.attribution.candidate_causes?.join('、') }}</p><p class="brief-meta">{{ citationDetail.attribution.model_id }} · {{ citationDetail.attribution.model_version }}</p><p class="brief-meta">输入截止 {{ mineTime(citationDetail.attribution.input_as_of) }}</p></template>
    </template>
  </section>
</section>
</template>

<style scoped>
.event-brief{color:#c7d9e9;font-size:13px;min-width:0}.brief-heading{display:flex;align-items:center;justify-content:space-between;gap:8px}.brief-heading h3,.brief-section h3{font-size:14px;font-weight:500;margin:0}.brief-state{border:1px solid #35536a;border-radius:4px;padding:1px 6px;font-size:12px;white-space:nowrap;color:#a6c5da}.brief-state.completed{color:#8cd4c1;border-color:#2e6057}.brief-state.failed,.brief-state.partial,.warning{color:#e6b978}.brief-version{font-size:12px;color:#9cc9dc;margin:10px 0 5px}.brief-meta{font-size:12px;line-height:1.7;color:#91abc2;margin:5px 0;overflow-wrap:anywhere}.brief-text{white-space:pre-wrap;line-height:1.9;margin:15px 0;overflow-wrap:anywhere}.inline-citation{background:none;color:#76d1e0;padding:0 2px;font-size:inherit;text-decoration:underline;text-underline-offset:3px}.brief-message{line-height:1.8;font-size:13px;color:#94b4ca;margin:13px 0}.brief-actions{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}.brief-button{padding:5px 9px;background:#16364b;border:1px solid #33566e;border-radius:4px;color:#acd4e5;font-size:12px;white-space:nowrap;cursor:pointer}.brief-button.primary{background:#1d6d84;border-color:#338ba0;color:#e1f8ff}.brief-button:disabled{opacity:.5;cursor:not-allowed}.brief-section{border-top:1px solid #29455b;padding-top:14px;margin-top:16px;min-width:0}.brief-steps{list-style:none;padding:0;margin:11px 0}.brief-steps li{border-left:2px solid #376076;padding:0 0 13px 10px;margin:0}.brief-steps li>div{display:flex;gap:8px;align-items:center;justify-content:space-between}.brief-steps strong{font-size:13px;font-weight:400}.step-state{font-size:12px;white-space:nowrap;color:#94b4ca}.step-state.success{color:#83d2bd}.step-state.error{color:#e6b978}.brief-steps p{font-size:12px;color:#91abc2;margin:5px 0;overflow-wrap:anywhere}.brief-steps small{font-size:12px;color:#7f9fb7}.brief-citations{list-style:none;margin:10px 0;padding:0}.brief-citations li{margin-bottom:6px}.brief-citations button{display:flex;align-items:start;gap:8px;padding:9px 8px;width:100%;text-align:left;background:#133247;color:#b4d8e8;border:1px solid #294d64;border-radius:4px;font-size:12px}.brief-citations b{font-weight:400;color:#7bd1df}.brief-citations span{flex:1;min-width:0;overflow-wrap:anywhere}.brief-citations small{display:block;font-size:12px;color:#88a8bf;margin-top:3px}.brief-citations i{font-style:normal;color:#72bbce}.brief-facts{margin:12px 0;font-size:12px}.brief-facts>div{display:grid;grid-template-columns:90px minmax(0,1fr);gap:8px;margin:7px 0}.brief-facts dt{color:#89a8bf}.brief-facts dd{margin:0;overflow-wrap:anywhere}.brief-table{overflow:auto;max-width:100%;margin-top:12px;max-height:280px}.brief-table table{font-size:12px;width:100%;border-collapse:collapse}.brief-table th,.brief-table td{white-space:nowrap;padding:8px;border-bottom:1px solid #29455b}.brief-table th{color:#8fb0c7;text-align:left}.brief-fragment{margin:12px 0;padding:9px;background:#102c41;border:1px solid #2b4a61;border-radius:4px}.brief-fragment p{white-space:pre-wrap;line-height:1.9}.brief-fragment small{display:block;color:#8baac1;font-size:12px;margin-top:7px}.citation-detail{padding:12px;background:#0c2438;border:1px solid #33566e;border-radius:5px}.citation-detail .brief-button{padding:3px 7px}
</style>
