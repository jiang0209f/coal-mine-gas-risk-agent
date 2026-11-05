<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { mineTime, queryString } from '../utils.mjs'
import { createRequestGuard, modelKey } from '../workspace.mjs'
import { detailRowLabel, evaluationReason, evaluationRequest, metricDisplay, metricNames } from '../evaluation.mjs'
import FrozenRecord from './FrozenRecord.vue'

const props = defineProps({ api: Function, safety: Boolean, session: Object, sourceId: String, points: Array, pointId: String, models: Array, selectedModel: String, reports: Array })
const emit = defineEmits(['created', 'refresh', 'error', 'notice'])
const form = ref({ start: '', end: '', point: '', model: '', stage: '', split: 'unclassified', tolerance: 0 })
const working = ref(false), loading = ref(false), selected = ref(null), manifest = ref(null), detail = ref(null), selectedRow = ref(null), metric = ref(''), variant = ref('primary'), detailLoading = ref(false)
const selectedObservation = ref(null), observationLoading = ref(false), observationId = ref('')
const reportRequests = createRequestGuard(), detailRequests = createRequestGuard(), observationRequests = createRequestGuard()
const scope = () => JSON.stringify([props.sourceId, props.session?.session_id])
const localInput = value => mineTime(value).replace(' ', 'T')
const methanePoints = computed(() => (props.points || []).filter(point => point.quantity === 'methane'))
const splitNames = { unclassified: '未标定', train: '训练集（人工声明）', validation: '验证集（人工声明）', test: '测试集（人工声明）' }
const view = computed(() => variant.value === 'primary' ? selected.value : selected.value?.calibration_views?.[variant.value] || selected.value)
const frozen = computed(() => manifest.value?.frozen_records || {})
const frozenScope = computed(() => selected.value?.scope_metadata || manifest.value?.scope_metadata || {})
const counts = computed(() => Object.fromEntries(['actual_events', 'alert_events', 'samples', 'tasks'].map(key => [key, frozen.value[key]?.length ?? selected.value?.sample_refs?.[key]?.length ?? 0])))
const rows = computed(() => detail.value?.rows || detail.value?.items || [])
const observationRefs = computed(() => {
  const records = [selectedRow.value?.record, ...snapshots.value.map(snapshot => snapshot.value)].filter(value => value && !Array.isArray(value))
  return [...new Map(records.flatMap(record => record.evidence_refs || []).filter(ref => ref?.kind === 'observation' || ref?.observation_id).map(ref => [ref.observation_id || ref.id, ref])).values()].filter(ref => ref.observation_id || ref.id)
})
const reportLabel = report => `${mineTime(report.generated_at, false)} · ${report.scope_metadata?.source_id || report.face_id} · ${report.scope_metadata?.session_id || '冻结清单'}`
const metricStatus = key => view.value?.metric_statuses?.[key] || selected.value?.metric_statuses?.[key] || {}
const missingName = key => ({ observed_event_evaluation: '未保存会话实测阈值事件评估记录', no_completed_forecast_runs: '没有已完成预测运行', no_mature_forecast_samples: '没有已成熟预测配对', no_forecast_runs: '没有预测运行记录', forecast_event_policy_not_supported: '当前预测预警口径未纳入事件评估', valid_monitoring_hours_excluding_suspected: '缺少剔除疑似标校后的有效时长', independent_actual_events: '尚未提供独立事件标注', dataset_split: '尚未核定数据分区', actual_events: '缺少实测事件', task_records: '缺少工具任务记录', no_methane_points: '当前范围没有甲烷测点' }[key] || '部分评估记录尚未齐备')
const snapshots = computed(() => {
  const row = selectedRow.value
  if (!row) return []
  const records = Array.isArray(row.snapshots) ? row.snapshots.map((value, index) => ({ title: `冻结记录${index + 1}`, value }))
    : row.snapshots && typeof row.snapshots === 'object' ? Object.entries(row.snapshots).filter(([, value]) => value && typeof value === 'object').map(([key, value]) => ({ title: { actual_event: '会话实测事件快照', alert_event: '预测事件快照', sample: '预测配对样本', task: '工具任务快照', advice: '建议版本快照', monitoring_interval: '监测支持区间' }[key] || '冻结事实', value }))
      : [{ title: '冻结事实', value: row.record || row.actual_event || row.alert_event || row }]
  const runIds = new Set(records.flatMap(({ value }) => [value?.run_id, value?.forecast_run_id]).filter(Boolean))
  const forecasts = (frozen.value.forecast_runs || []).filter(run => runIds.has(run.run_id))
  return [...records, ...forecasts.map(value => ({ title: '冻结预测运行', value }))]
})
function clearSelection() {
  reportRequests.invalidate(); detailRequests.invalidate(); observationRequests.invalidate(); selected.value = null; manifest.value = null; detail.value = null; selectedRow.value = null; selectedObservation.value = null; observationLoading.value = false; metric.value = ''; variant.value = 'primary'; loading.value = false; detailLoading.value = false; working.value = false
}
function resetForm() {
  form.value = { start: props.session ? localInput(props.session.start_as_of) : '', end: props.session ? localInput(props.session.as_of) : '',
    point: methanePoints.value.some(point => point.point_id === props.pointId) ? props.pointId : '', model: props.selectedModel || '', stage: '', split: 'unclassified', tolerance: 0 }
}
watch(scope, () => { clearSelection(); resetForm() }, { immediate: true })
watch(() => props.session?.as_of, (value, previous) => { if (!form.value.end || (previous && form.value.end === localInput(previous))) form.value.end = value ? localInput(value) : '' })
watch(() => props.selectedModel, value => { if (!form.value.model) form.value.model = value || '' })
watch(selectedRow, () => { observationRequests.invalidate(); selectedObservation.value = null; observationLoading.value = false; observationId.value = '' })
onBeforeUnmount(clearSelection)
async function selectReport(report) {
  const request = reportRequests.begin(), requestedScope = scope()
  detailRequests.invalidate(); selected.value = null; manifest.value = null; detail.value = null; selectedRow.value = null; metric.value = ''; variant.value = 'primary'; loading.value = true
  try {
    const [saved, frozenManifest] = await Promise.all([props.api(`/evaluations/${encodeURIComponent(report.evaluation_id)}`), props.api(`/evaluations/${encodeURIComponent(report.evaluation_id)}/manifest`)])
    if (!reportRequests.isCurrent(request) || requestedScope !== scope()) return
    selected.value = saved; manifest.value = frozenManifest
  } catch (error) { if (reportRequests.isCurrent(request) && requestedScope === scope()) emit('error', error.message) }
  finally { if (reportRequests.isCurrent(request)) loading.value = false }
}
async function createEvaluation() {
  const requestedScope = scope(); working.value = true
  try {
    const body = evaluationRequest({ session: props.session, startTime: form.value.start, endTime: form.value.end,
      pointIds: form.value.point ? [form.value.point] : methanePoints.value.map(point => point.point_id),
      model: (props.models || []).find(item => modelKey(item) === form.value.model), predictionStage: form.value.stage,
      datasetSplit: form.value.split, toleranceSeconds: form.value.tolerance })
    const result = await props.api('/evaluations/from-session', { method: 'POST', body })
    if (requestedScope !== scope()) return
    emit('created', result); await selectReport(result); emit('notice', '会话记录已采集，评估报告已冻结。')
  } catch (error) { if (requestedScope === scope()) emit('error', error.message) }
  finally { if (requestedScope === scope()) working.value = false }
}
async function loadDetails(key, offset = 0) {
  if (!selected.value) return
  const id = selected.value.evaluation_id, requestedScope = scope(), request = detailRequests.begin()
  metric.value = key; detail.value = null; selectedRow.value = null; detailLoading.value = true
  try {
    const result = await props.api(`/evaluations/${encodeURIComponent(id)}/metrics/${encodeURIComponent(key)}?${queryString({ variant: variant.value, offset, limit: 50 })}`)
    if (!detailRequests.isCurrent(request) || requestedScope !== scope() || selected.value?.evaluation_id !== id) return
    detail.value = result
  } catch (error) { if (detailRequests.isCurrent(request) && requestedScope === scope()) emit('error', error.message) }
  finally { if (detailRequests.isCurrent(request)) detailLoading.value = false }
}
function changeVariant() { detail.value = null; selectedRow.value = null; if (metric.value) void loadDetails(metric.value) }
async function openObservations(event) {
  if (!event.target.open || !observationRefs.value.length) return
  observationId.value = observationRefs.value[0].observation_id || observationRefs.value[0].id
  await loadObservation()
}
async function loadObservation() {
  if (!selected.value || !selectedRow.value || !observationId.value) return
  const id = selected.value.evaluation_id, row = selectedRow.value, requestedScope = scope(), request = observationRequests.begin()
  selectedObservation.value = null; observationLoading.value = true
  try {
    const result = await props.api(`/evaluations/${encodeURIComponent(id)}/records/observations/${encodeURIComponent(observationId.value)}`)
    if (!observationRequests.isCurrent(request) || requestedScope !== scope() || selectedRow.value !== row) return
    if (result.evaluation_id !== id || result.records_sha256 !== selected.value.records_sha256) throw new Error('冻结观测与报告摘要不一致。')
    selectedObservation.value = result.record
  } catch (error) { if (observationRequests.isCurrent(request) && requestedScope === scope()) emit('error', error.message) }
  finally { if (observationRequests.isCurrent(request)) observationLoading.value = false }
}
async function downloadReport() {
  if (!selected.value) return
  const id = selected.value.evaluation_id, requestedScope = scope()
  try {
    const blob = await props.api(`/evaluations/${encodeURIComponent(id)}/export`, { responseType: 'blob' })
    if (requestedScope !== scope() || selected.value?.evaluation_id !== id) return
    const url = URL.createObjectURL(blob), anchor = document.createElement('a')
    anchor.href = url; anchor.download = `瓦斯评估-${id}.json`; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 2000)
  } catch (error) { if (requestedScope === scope()) emit('error', error.message) }
}
</script>

<template>
  <div class="evaluation-workspace">
    <section v-if="!safety" class="panel empty-state"><p>评估报告由安全管理人员查看。</p></section>
    <template v-else>
      <section class="panel"><div class="section-heading"><div><h2>会话评估</h2></div></div>
        <p v-if="!session" class="muted">选择会话后创建评估。</p>
        <form v-else @submit.prevent="createEvaluation"><div class="form-row"><label>评估开始（UTC+8）<input v-model="form.start" type="datetime-local" step="1" required></label><label>评估结束（UTC+8）<input v-model="form.end" type="datetime-local" step="1" required></label></div>
          <div class="form-row"><label>评估测点<select v-model="form.point"><option value="">全部甲烷测点</option><option v-for="point in methanePoints" :key="point.point_id" :value="point.point_id">{{ point.name || point.point_id }}</option></select></label><label>评估模型<select v-model="form.model"><option value="">按已保存运行确定</option><option v-for="model in models || []" :key="modelKey(model)" :value="modelKey(model)">{{ model.name || model.model_id }} · {{ model.version }}</option></select></label></div>
          <details class="evaluation-settings"><summary>评估口径</summary><div class="form-row"><label>数据分区<select v-model="form.split"><option v-for="(label,key) in splitNames" :key="key" :value="key">{{ label }}</option></select></label><label>事件匹配容差（秒）<input v-model.number="form.tolerance" type="number" min="0" step="1"></label></div><label>物理预测输出<select v-model="form.stage"><option value="">按已保存运行确定</option><option value="baseline_persistence">持久性基线</option><option value="fedformer">FEDformer</option><option value="residual_corrected">残差修正预测</option></select></label></details>
          <p class="muted">数据截止 {{ mineTime(session.as_of) }}。实测阈值事件与预测预警分别采集，多测点时长按监测点小时相加。</p><button class="primary" :disabled="working || loading || !methanePoints.length">{{ working ? '正在采集与冻结' : '冻结当前会话评估' }}</button>
        </form>
      </section>
      <section class="panel"><div class="section-heading"><div><h2>冻结评估报告</h2></div><button class="secondary small" :disabled="working || loading" @click="emit('refresh')">刷新报告</button></div><div v-if="!reports?.length" class="empty-state"><p>暂无冻结评估报告</p></div><button v-for="report in reports || []" :key="report.evaluation_id" class="report-choice" :class="{ active: selected?.evaluation_id === report.evaluation_id }" @click="selectReport(report)"><strong>{{ reportLabel(report) }}</strong><small>报告 {{ report.evaluation_id }}</small></button><p v-if="loading" class="muted">正在读取冻结报告。</p></section>
      <section v-if="selected && manifest" class="panel" role="region" aria-label="冻结评估详情"><div class="section-heading"><div><h2>评估范围与指标</h2><p>{{ mineTime(selected.generated_at) }} · {{ splitNames[manifest.dataset_split] || manifest.dataset_split }}</p></div><button class="secondary small" @click="downloadReport">下载冻结报告</button></div>
        <dl class="evaluation-facts"><div><dt>来源 / 会话</dt><dd>{{ frozenScope.source_id || '见冻结清单' }} / {{ frozenScope.session_id || '未记录会话' }}</dd></div><div><dt>评估区间</dt><dd>{{ mineTime(frozenScope.start_at) }} — {{ mineTime(frozenScope.end_at) }}</dd></div><div><dt>测点</dt><dd>{{ frozenScope.point_ids?.join('、') || '见冻结清单' }}</dd></div><div><dt>模型 / 规则</dt><dd>{{ manifest.model_version }} / {{ manifest.rule_version }}</dd></div><div><dt>有效监测点小时</dt><dd>{{ metricDisplay(frozen.valid_monitoring_hours ?? selected.collection_summary?.valid_monitoring_hours) }}</dd></div><div><dt>采集条数</dt><dd>{{ counts.actual_events }}条会话实测阈值事件 · {{ counts.alert_events }}个预测事件 · {{ counts.samples }}个配对样本 · {{ counts.tasks }}个工具任务</dd></div></dl>
        <p class="muted">实测事件按冻结规则阈值识别；样本按物理预测区间与已到达的有效实测统计值配对，汇总方式见冻结记录。</p><p v-for="field in [...new Set([...(selected.missing_fields || []), ...(view?.missing_fields || [])])]" :key="field" class="hint amber-text">{{ missingName(field) }}</p>
        <label class="evaluation-view">标校统计口径<select v-model="variant" @change="changeVariant"><option value="primary">主报告</option><option value="all_events">全部记录</option><option value="excluding_suspected">剔除疑似标校</option></select></label>
        <div class="table-scroll"><table class="metric-table"><thead><tr><th>指标</th><th>结果</th><th>分子 / 分母</th><th>依据</th></tr></thead><tbody><tr v-for="(value,key) in view?.metric_values || {}" :key="key"><td>{{ metricNames[key] || key }}</td><td>{{ metricDisplay(value) }}<small v-if="evaluationReason(metricStatus(key))">{{ evaluationReason(metricStatus(key)) }}</small></td><td>{{ metricDisplay(view?.numerators?.[key]) }} / {{ metricDisplay(view?.denominators?.[key]) }}</td><td><button class="ghost small" :aria-label="`查看${metricNames[key] || key}明细`" @click="loadDetails(key)">查看明细</button></td></tr></tbody></table></div>
        <details class="evaluation-settings"><summary>冻结清单与校验摘要</summary><dl class="evaluation-facts"><div><dt>清单编号</dt><dd>{{ selected.manifest_id }}</dd></div><div><dt>记录摘要</dt><dd>{{ selected.records_sha256 }}</dd></div><div><dt>清单摘要</dt><dd>{{ selected.manifest_sha256 }}</dd></div><div><dt>报告摘要</dt><dd>{{ selected.report_sha256 }}</dd></div><div v-if="manifest.export_records_sha256"><dt>导出记录摘要</dt><dd>{{ manifest.export_records_sha256 }}</dd></div><div><dt>数据集引用</dt><dd>{{ manifest.dataset_refs?.join('、') || '见采集范围' }}</dd></div><div><dt>加工版本</dt><dd>{{ manifest.preprocess_version }}</dd></div></dl></details>
      </section>
      <section v-if="metric" class="panel" role="region" aria-label="指标明细"><div class="section-heading"><div><h2>{{ metricNames[metric] || metric }}明细</h2></div></div><p v-if="detailLoading" class="muted">正在读取指标明细。</p><template v-if="detail"><p>{{ metricDisplay(detail.numerator) }} / {{ metricDisplay(detail.denominator) }} = {{ metricDisplay(detail.value) }}</p><p v-if="evaluationReason(detail)" class="hint amber-text">{{ evaluationReason(detail) }}</p><div class="detail-pagination"><span>共{{ detail.total }}条</span><button class="secondary small" :disabled="!detail.offset || detailLoading" @click="loadDetails(metric, Math.max(0,detail.offset - detail.limit))">上一页</button><button class="secondary small" :disabled="detail.offset + detail.limit >= detail.total || detailLoading" @click="loadDetails(metric, detail.offset + detail.limit)">下一页</button></div><ol class="evaluation-rows"><li v-for="(row,index) in rows" :key="`${detail.offset}-${index}`"><div><strong>{{ detailRowLabel(row) }}</strong><small>{{ row.included === false ? '未计入当前口径' : row.matched === true ? '已匹配' : row.kind === 'event_match' ? '一对一匹配' : '冻结明细' }}</small><span v-if="row.absolute_error !== undefined">绝对误差 {{ metricDisplay(row.absolute_error) }}</span><span v-if="row.hours !== undefined">监测点小时 {{ metricDisplay(row.hours) }}</span><small v-if="row.numerator_contribution !== undefined || row.denominator_contribution !== undefined">分子贡献 {{ metricDisplay(row.numerator_contribution) }} · 分母贡献 {{ metricDisplay(row.denominator_contribution) }}</small></div><button class="ghost small" :aria-label="`查看冻结记录 ${detail.offset + index + 1}`" @click="selectedRow = row">查看冻结记录</button></li></ol><p v-if="!rows.length" class="muted">当前口径没有明细记录。</p></template></section>
      <section v-if="selectedRow" class="panel frozen-detail" role="region" aria-label="冻结记录"><div class="section-heading"><div><h2>{{ detailRowLabel(selectedRow) }}</h2><span class="tag teal">冻结快照</span></div><button class="ghost small" @click="selectedRow = null">关闭记录</button></div><article v-for="(snapshot,index) in snapshots" :key="index"><h3>{{ snapshot.title }}</h3><FrozenRecord :record="snapshot.value" /></article><details v-if="observationRefs.length" :key="detailRowLabel(selectedRow)" class="evaluation-settings" @toggle="openObservations"><summary>冻结观测 · {{ observationRefs.length }}条</summary><label>冻结观测选择<select v-model="observationId" @change="loadObservation"><option v-for="(reference,index) in observationRefs" :key="reference.observation_id || reference.id" :value="reference.observation_id || reference.id">记录{{ index + 1 }} · {{ reference.point_id || reference.observation_id || reference.id }}{{ reference.row_number ? ' · 原始行' + reference.row_number : '' }}</option></select></label><p v-if="observationLoading" class="muted">正在读取冻结观测。</p><FrozenRecord v-if="selectedObservation" :record="selectedObservation" /></details></section>
    </template>
  </div>
</template>

<style scoped>
.evaluation-workspace{grid-template-columns:minmax(0,1fr)}.evaluation-workspace>.panel{min-width:0;max-width:100%}.evaluation-workspace form,.evaluation-workspace .table-scroll{min-width:0;max-width:100%}.evaluation-workspace .form-row>*{min-width:0;max-width:100%}.evaluation-workspace input[type="datetime-local"]{min-width:0;max-width:100%;width:100%}
.evaluation-workspace{display:grid;gap:18px;min-width:0}.evaluation-settings{margin:15px 0}.evaluation-settings>summary{cursor:pointer;color:#8acbdb;font-size:13px}.evaluation-settings[open]>summary{margin-bottom:13px}.report-choice{display:block;width:100%;text-align:left;background:#102a40;border:1px solid #28485f;color:#c9deeb;border-radius:4px;padding:12px;margin:9px 0;min-width:0}.report-choice.active{border-color:#4397ac}.report-choice strong{font-size:13px;font-weight:500;display:block;overflow-wrap:anywhere}.report-choice small{display:block;font-size:12px;color:#89a7bb;margin-top:4px;overflow-wrap:anywhere}.evaluation-facts{display:grid;gap:10px;font-size:13px;margin:17px 0}.evaluation-facts>div{display:grid;grid-template-columns:125px minmax(0,1fr);gap:10px;line-height:1.65}.evaluation-facts dt{color:#8eaec3}.evaluation-facts dd{margin:0;overflow-wrap:anywhere}.evaluation-view{display:flex;align-items:center;gap:12px;max-width:360px;white-space:nowrap}.evaluation-view select{margin:0;flex:1;min-width:0}.metric-table{min-width:510px}.metric-table td{white-space:normal}.metric-table td>small{display:block;margin-top:5px;color:#dcba83}.detail-pagination{display:flex;align-items:center;gap:9px;margin:15px 0;flex-wrap:wrap}.detail-pagination span{font-size:12px;color:#8faec1;margin-right:auto}.evaluation-rows{list-style:none;padding:0;margin:0}.evaluation-rows li{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:11px 0;border-bottom:1px solid #27455a}.evaluation-rows li>div{min-width:0}.evaluation-rows strong{display:block;font-size:13px;font-weight:500;overflow-wrap:anywhere}.evaluation-rows small,.evaluation-rows span{display:block;font-size:12px;color:#8eaec3;margin-top:4px}.evaluation-rows button{flex-shrink:0}.frozen-detail article{border-top:1px solid #29475c;padding-top:14px;margin-top:16px}.frozen-detail h3{font-size:13px;color:#9ed4df;font-weight:500}@media(max-width:760px){.evaluation-workspace{gap:12px}.evaluation-facts>div{grid-template-columns:93px minmax(0,1fr);gap:8px}.evaluation-rows li{align-items:flex-start}.evaluation-workspace .section-heading>div{min-width:0}.evaluation-workspace .section-heading h2,.evaluation-workspace .section-heading p{overflow-wrap:anywhere}.evaluation-workspace .section-heading>button{flex-shrink:0;padding:7px 9px}.evaluation-rows button{padding:5px 7px}.evaluation-view{max-width:none}}
</style>
