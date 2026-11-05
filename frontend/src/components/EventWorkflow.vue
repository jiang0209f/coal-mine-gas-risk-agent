<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { assessmentLabel, canExecuteControlTask, canWorkflowReview, createEventWorkflowClient, isControlTaskOverdue, parseReferenceIds, planDeadlines, shouldPollInvestigation, unwrapInvestigation, workflowState } from '../event-workflow.mjs'
import { mineTime } from '../utils.mjs'
import { referenceLocation } from '../knowledge.mjs'
import { toolName, toolState } from '../assistant.mjs'

const props = defineProps({ api: Function, session: Object, event: Object, mode: String, roles: { type: Array, default: () => [] } })
const report = ref(null), history = ref([]), control = ref({ measures: [], plans: [], tasks: [], experiences: [], effect: null })
const loading = ref(false), working = ref(false), error = ref(''), notice = ref(''), pollPaused = ref(false)
const reportNote = ref(''), measureReferences = ref(''), reviewNotes = ref({}), taskNotes = ref({}), completionNote = ref('')
const selectedPlanId = ref(''), planDetail = ref(null), citation = ref(null), citationLoading = ref(false)
const tierInputs = ref({}), deadlineInputs = ref({}), assignedRoleInputs = ref({})
let generation = 0, loadSequence = 0, timer, pollCount = 0
const stateNames = { not_started: '尚未开始', queued: '排队中', running: '排查中', draft: '待审核', pending: '待审核', pending_review: '待审核', approved: '已审核', rejected: '已退回', failed: '未完成', partial: '部分完成', assigned: '待执行', ready: '待执行', new: '待执行', executing: '执行中', in_progress: '执行中', executed: '已执行', completed: '已完成', verified: '已核验', cancelled: '已取消', effective: '见效', ineffective: '未见效', insufficient_data: '数据不足', escalated: '已建议升级' }
const roleNames = { ventilation: '通风人员', gas_governance: '瓦斯治理人员', safety_management: '安全管理人员' }
const causeNames = { ventilation_abnormal: '通风系统异常', gas_extraction_inadequate: '抽采不足', goaf_gas_emission: '采空区瓦斯涌出', adjacent_seam_gas: '邻近层瓦斯涌出', coal_cutting: '割煤速度快' }
const stepNames = { get_event_evidence: '读取事件依据', compare_points: '比较邻近测点', get_condition_timeline: '查看工况时间线', get_face_statistics: '工作面工况统计', find_similar_events: '查找相似事件' }
const missingNames = { cause_pending_verify: '原因尚待核实', approved_applicable_measures: '缺少已审核适用措施', all_tasks_executed: '执行任务尚未全部完成', post_execution_window: '执行后的5分钟监测窗口尚未完整', valid_window_samples_or_threshold: '有效监测样本或核验阈值不足' }
const reportId = computed(() => report.value?.report_id || report.value?.investigation_id || report.value?.id)
const safety = computed(() => props.roles.includes('safety_management'))
const technician = computed(() => props.roles.some(role => roleNames[role]))
const reportCurrent = computed(() => !report.value?.event_revision || !props.event?.event_revision || report.value.event_revision === props.event.event_revision)
const selectedPlan = computed(() => planDetail.value?.plan || control.value.plans.find(item => item.plan_id === selectedPlanId.value))
const tasks = computed(() => planDetail.value?.tasks || control.value.tasks.filter(item => !item.plan_id || item.plan_id === selectedPlanId.value))
const effect = computed(() => {
  if (planDetail.value && Object.hasOwn(planDetail.value, 'effect')) return planDetail.value.effect
  const value = control.value.effect
  return value?.plan_id && value.plan_id !== selectedPlanId.value ? null : value
})
const experiences = computed(() => control.value.experiences || [])
const hypotheses = computed(() => report.value?.hypotheses || [])
const steps = computed(() => report.value?.investigation_steps || report.value?.task?.tool_steps || report.value?.task?.tool_calls || [])
const canVerify = computed(() => !working.value && technician.value && selectedPlan.value && tasks.value.length > 0 && tasks.value.every(item => ['executed', 'completed', 'verified'].includes(workflowState(item))))
const scope = () => JSON.stringify([props.session?.session_id, props.session?.source_id, props.event?.event_id, props.event?.event_revision, props.mode])
const client = () => createEventWorkflowClient(props.api, props.session?.session_id, props.event?.event_id)
const stateLabel = item => stateNames[workflowState(item)] || '待核对'
const itemText = item => typeof item === 'string' ? item : item?.summary || item?.content || item?.text || item?.description || item?.rationale || item?.reason || item?.title || item?.cause || ''
const identity = item => item?.measure_id || item?.experience_id || item?.plan_id || item?.report_id || item?.id
const refsFor = item => (item?.evidence_refs || item?.references || item?.reference_ids || (item?.reference_id ? [item] : [])).map(reference => typeof reference === 'string' ? { reference_id: reference } : reference)
const refLabel = item => item?.label || item?.title || item?.reference_id || item?.observation_id || item?.run_id || item?.event_id || '依据记录'
const reviewAllowed = item => canWorkflowReview(item, props.roles, working.value)
const taskAllowed = task => canExecuteControlTask(task, props.roles, taskNotes.value[task.task_id] || '', workflowState(selectedPlan.value), working.value)
const notesKey = (kind, item) => `${kind}:${identity(item)}`
const measureRole = item => assignedRoleInputs.value[item.measure_id] ?? (roleNames[item.assigned_role] ? item.assigned_role : '')
const stepLabel = step => step.title || step.label || stepNames[step.tool || step.name] || toolName(step.tool || step.name)
function metricFacts(value) {
  const labels = { peak: '峰值', peak_value: '峰值', peak_offset_seconds: '峰值相对时间 / 秒', rise_offset_seconds: '开始上升相对时间 / 秒', rise_lag_seconds: '上升滞后 / 秒', peak_lag_seconds: '峰值滞后 / 秒', count: '记录数', valid_count: '有效记录数', quantity: '监测量', unit: '单位', minimum: '最低值', maximum: '最高值', mean: '平均值', last: '最新值', first: '起始值', delta: '变化量', change: '变化量', state: '状态', cutting_state: '割煤状态', speed: '速度', position: '位置', wind_speed: '风速', left_current: '左截割电流', right_current: '右截割电流', value: '数值', total: '总记录数', high_count: '高值记录数', high_ratio: '高值比例', threshold: '阈值', similarity: '相似度', high_samples: '高值样本', high_fraction: '高值比例', sample_count: '样本数', window_seconds: '统计窗口 / 秒' }
  const names = { methane: '甲烷', wind_speed: '风速', cutting: '割煤', idle: '空载', running: '运行', stopped: '停止', unknown: '待核实' }
  const result = []
  function walk(item, prefix = '') {
    if (result.length >= 28 || !item || typeof item !== 'object') return
    if (Array.isArray(item)) { item.slice(0, 6).forEach((row, index) => walk(row, `${prefix}${row?.point_id || row?.point || index + 1} · `)); return }
    for (const [key, child] of Object.entries(item)) {
      if (child && typeof child === 'object') walk(child, prefix)
      else if (labels[key] && child !== undefined && child !== null) result.push({ label: `${prefix}${labels[key]}`, value: stateNames[child] || names[child] || child })
      if (result.length >= 28) break
    }
  }
  walk(value)
  return result
}

function stopPoll() { clearTimeout(timer); timer = undefined }
function schedulePoll() {
  stopPoll()
  if (!shouldPollInvestigation(report.value)) return
  if (++pollCount > 90) { pollPaused.value = true; return }
  timer = setTimeout(() => refresh(true), 2000)
}
async function refresh(polling = false) {
  if (!props.session || !props.event || !props.api) return
  const current = generation, sequence = ++loadSequence, api = client()
  if (!polling) loading.value = true
  error.value = ''
  try {
    const results = await Promise.allSettled([api.loadInvestigation(), props.mode === 'control' ? api.loadControl() : Promise.resolve(null)])
    if (current !== generation || sequence !== loadSequence) return
    if (results[0].status === 'fulfilled') {
      report.value = unwrapInvestigation(results[0].value)
      history.value = results[0].value?.history || []
    } else if (results[0].reason.status !== 404) throw results[0].reason
    if (results[1].status === 'fulfilled' && results[1].value) {
      const packet = results[1].value
      control.value = { ...packet, measures: packet.measures || [], plans: packet.plans || [], tasks: packet.tasks || [], experiences: packet.experiences || [] }
      if (!control.value.plans.some(item => item.plan_id === selectedPlanId.value)) selectedPlanId.value = control.value.plans[0]?.plan_id || ''
      if (selectedPlanId.value) {
        const detail = await api.getPlan(selectedPlanId.value)
        if (current !== generation || sequence !== loadSequence) return
        planDetail.value = detail?.plan ? detail : { plan: detail }
      }
    } else if (results[1].status === 'rejected') throw results[1].reason
    schedulePoll()
  } catch (caught) {
    if (current === generation && sequence === loadSequence) { error.value = caught.message; stopPoll(); pollPaused.value = shouldPollInvestigation(report.value) }
  } finally { if (current === generation && sequence === loadSequence) loading.value = false }
}
async function act(callback, message) {
  if (working.value) return
  const current = generation, api = client()
  working.value = true; error.value = ''; notice.value = ''
  try {
    await callback(api)
    if (current !== generation) return
    notice.value = message || '已保存'
    await refresh()
  } catch (caught) { if (current === generation) error.value = caught.message }
  finally { if (current === generation) working.value = false }
}
function startInvestigation() {
  pollCount = 0; pollPaused.value = false
  return act(async api => { report.value = unwrapInvestigation(await api.startInvestigation(Boolean(reportId.value))); reportNote.value = '' }, '已提交排查任务')
}
function createControlPlan() {
  return act(async api => {
    const value = await api.createPlan(reportId.value)
    selectedPlanId.value = value?.plan_id || value?.plan?.plan_id || ''
    planDetail.value = null; deadlineInputs.value = {}
  }, '计划已生成，等待审核')
}
async function selectHistory(event) {
  const current = generation
  try {
    const value = await client().getInvestigation(event.target.value)
    if (current === generation) report.value = unwrapInvestigation(value)
  } catch (caught) { if (current === generation) error.value = caught.message }
}
async function selectPlan(event) {
  const current = generation, id = event.target.value
  loadSequence++; loading.value = false
  selectedPlanId.value = id; planDetail.value = null
  deadlineInputs.value = {}
  try {
    const value = await client().getPlan(id)
    if (current === generation && id === selectedPlanId.value) planDetail.value = value?.plan ? value : { plan: value }
  } catch (caught) { if (current === generation) error.value = caught.message }
}
async function downloadReport() {
  await act(async api => {
    const blob = await api.downloadReport(reportId.value)
    const url = URL.createObjectURL(blob), link = document.createElement('a')
    link.href = url; link.download = `瓦斯事件排查报告-${reportId.value}.docx`; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }, '报告已导出')
}
async function openCitation(reference) {
  const current = generation
  citation.value = reference; citationLoading.value = false
  if (reference.kind === 'tool_result') {
    const step = steps.value.find(item => item.step_id === reference.reference_id)
    citation.value = { ...reference, title: step ? stepLabel(step) : '工具依据', metrics: step?.metrics, evidence_refs: reference.evidence_refs || step?.evidence_refs || [] }
    return
  }
  if (!reference.reference_id) return
  citationLoading.value = true
  try {
    const fragment = await props.api(`/references/fragments/${encodeURIComponent(reference.reference_id)}`)
    if (current === generation) citation.value = { ...reference, ...(fragment.fragment || fragment) }
  } catch (caught) { if (current === generation) error.value = caught.message }
  finally { if (current === generation) citationLoading.value = false }
}
function review(kind, item, decision) {
  const key = notesKey(kind, item), note = reviewNotes.value[key] || ''
  const methods = { measure: 'reviewMeasure', plan: 'reviewPlan', experience: 'reviewExperience' }
  return act(async api => {
    const extra = kind === 'measure' ? Number(tierInputs.value[item.measure_id] ?? item.tier ?? 1)
      : kind === 'plan' ? planDeadlines(item.items, deadlineInputs.value) : undefined
    if (kind === 'measure') await api.reviewMeasure(identity(item), decision, note, extra, measureRole(item) || undefined)
    else await api[methods[kind]](identity(item), decision, note, extra)
    reviewNotes.value[key] = ''
  }, decision === 'approve' ? '审核已通过' : '已退回补充')
}
const effectFacts = computed(() => {
  const labels = { status: '核验状态', state: '核验状态', before_maximum: '处置前最高值', after_maximum: '处置后最高值', baseline_maximum: '处置前最高值', reduction: '降低量', reduction_ratio: '降低比例', before_count: '处置前记录数', after_count: '处置后记录数', sample_count: '有效记录数', threshold: '核验阈值', observation_count: '监测记录数', risk_reduced: '风险是否降低', effective: '是否达到目标', verified_at: '核验时点' }
  return Object.entries(effect.value || {}).filter(([key, value]) => labels[key] && value !== null && typeof value !== 'object').map(([key, value]) => ({ label: labels[key], value: key.endsWith('_at') ? mineTime(value) : typeof value === 'boolean' ? value ? '是' : '否' : stateNames[value] || value }))
})
watch(scope, () => {
  generation++; loadSequence++; stopPoll(); pollCount = 0
  report.value = null; history.value = []; control.value = { measures: [], plans: [], tasks: [], experiences: [], effect: null }
  selectedPlanId.value = ''; planDetail.value = null; citation.value = null; reportNote.value = ''; completionNote.value = ''
  reviewNotes.value = {}; taskNotes.value = {}; error.value = ''; notice.value = ''; working.value = false; loading.value = false; pollPaused.value = false
  tierInputs.value = {}; deadlineInputs.value = {}; assignedRoleInputs.value = {}
  refresh()
}, { immediate: true })
watch(() => props.session?.as_of, (value, previous) => { if (value && previous && value !== previous) refresh() })
onBeforeUnmount(() => { generation++; stopPoll() })
</script>

<template>
  <section class="event-workflow" :aria-label="mode === 'control' ? '事件管控' : '事件溯源'" aria-live="polite">
    <div class="workflow-head"><h3>{{ mode === 'control' ? '管控闭环' : '事件排查' }}</h3><button class="workflow-button" :disabled="loading || working" @click="refresh()">刷新</button></div>
    <p v-if="loading" class="workflow-muted" role="status">正在读取事件记录…</p>
    <p v-if="error" class="workflow-warning" role="alert">{{ error }}</p><p v-if="notice" class="workflow-notice" role="status">{{ notice }}</p>
    <p v-if="pollPaused" class="workflow-muted">排查仍在后台执行，可刷新查看。</p>

    <template v-if="mode === 'investigation'">
      <div class="workflow-actions"><span :class="['workflow-state', workflowState(report)]">{{ stateLabel(report) }}</span><button class="workflow-button primary" :disabled="loading || working || shouldPollInvestigation(report)" @click="startInvestigation">{{ report ? '重新排查' : '启动排查' }}</button><button v-if="reportId && !shouldPollInvestigation(report) && workflowState(report) !== 'failed'" class="workflow-button" :disabled="working" @click="downloadReport">导出报告</button></div>
      <p v-if="report?.model || report?.task?.model" class="workflow-muted">{{ report.model || report.task.model }}</p>
      <p v-if="!reportCurrent" class="workflow-warning">事件版本已更新，请重新排查。</p>
      <label v-if="history.length > 1" class="workflow-label">排查记录<select :value="reportId" @change="selectHistory"><option v-for="item in history" :key="item.report_id || item.investigation_id" :value="item.report_id || item.investigation_id">{{ mineTime(item.created_at || item.as_of) }} · {{ stateLabel(item) }}</option></select></label>
      <p v-if="!report && !loading" class="workflow-muted">当前事件尚无排查报告。</p>
      <p v-if="report?.error_message || report?.error" class="workflow-warning">{{ itemText(report.error_message || report.error) || '排查未完成，请核对运行状态。' }}</p>
      <section v-if="steps.length" class="workflow-section" aria-label="排查过程"><h4>排查过程</h4><ol class="workflow-steps"><li v-for="(step,index) in steps" :key="step.call_id || step.step_id || index"><div><strong>{{ stepLabel(step) }}</strong><span>{{ stateNames[step.status || step.state] || toolState(step.status || step.state) }}</span></div><p v-if="itemText(step)">{{ itemText(step) }}</p><small v-if="Number.isFinite(step.elapsed_ms)">{{ step.elapsed_ms }} ms</small><details v-if="metricFacts(step.metrics).length"><summary>查看指标</summary><dl class="workflow-facts"><div v-for="(fact,factIndex) in metricFacts(step.metrics)" :key="factIndex"><dt>{{ fact.label }}</dt><dd>{{ fact.value }}</dd></div></dl></details></li></ol></section>
      <section v-if="report?.event_overview" class="workflow-section"><h4>事件概况</h4><p v-if="itemText(report.event_overview)">{{ itemText(report.event_overview) }}</p><dl v-else class="workflow-facts"><div v-if="report.event_overview.point_id"><dt>测点</dt><dd>{{ report.event_overview.point_id }}</dd></div><div v-if="report.event_overview.start_at"><dt>开始时间</dt><dd>{{ mineTime(report.event_overview.start_at) }}</dd></div><div v-if="report.event_overview.alert_level"><dt>预警等级</dt><dd>{{ { red:'红色', orange:'橙色', yellow:'黄色', blue:'蓝色' }[report.event_overview.alert_level] || '待核对' }}</dd></div></dl></section>
      <section v-if="hypotheses.length" class="workflow-section" aria-label="原因排查"><h4>原因排查</h4><article v-for="(hypothesis,index) in hypotheses" :key="hypothesis.cause_id || index" class="workflow-card"><div class="workflow-head"><strong>{{ hypothesis.cause }}</strong><span :class="['workflow-state', hypothesis.support_status || hypothesis.assessment]">{{ assessmentLabel(hypothesis) }}</span></div><p>{{ hypothesis.rationale || hypothesis.reason || '尚无判定说明' }}</p><div v-if="refsFor(hypothesis).length" class="workflow-citations"><button v-for="(reference,refIndex) in refsFor(hypothesis)" :key="reference.reference_id || refIndex" class="workflow-link" @click="openCitation(reference)">{{ refLabel(reference) }}</button></div></article></section>
      <section v-if="report?.conclusion || report?.summary" class="workflow-section"><h4>排查结论</h4><p>{{ itemText(report.conclusion || report.summary) }}</p></section>
      <section v-if="report?.supported_causes?.length" class="workflow-section"><h4>支持的候选原因</h4><ul><li v-for="(item,index) in report.supported_causes" :key="index">{{ itemText(item) }}</li></ul></section>
      <section v-if="report?.excluded_items?.length" class="workflow-section"><h4>排除项</h4><ul><li v-for="(item,index) in report.excluded_items" :key="index">{{ itemText(item) }}</li></ul></section>
      <section v-if="report?.pending_items?.length" class="workflow-section"><h4>待核查事项</h4><ul><li v-for="(item,index) in report.pending_items" :key="index">{{ itemText(item) }}</li></ul></section>
      <section v-if="reviewAllowed(report)" class="workflow-section" aria-label="排查报告人工审核"><h4>报告审核</h4><label class="workflow-label">审核意见<textarea v-model="reportNote" rows="3" aria-label="排查报告审核意见"></textarea></label><div class="workflow-actions"><button class="workflow-button primary" :disabled="working || !reportNote.trim() || !reportCurrent" @click="act(async api => { await api.reviewInvestigation(reportId, 'approve', reportNote); reportNote = '' }, '报告审核已通过')">审核通过</button><button class="workflow-button" :disabled="working || !reportNote.trim()" @click="act(async api => { await api.reviewInvestigation(reportId, 'reject', reportNote); reportNote = '' }, '报告已退回')">退回补充</button></div></section>
      <p v-else-if="report?.review?.note || report?.review_note" class="workflow-muted">审核意见：{{ report.review?.note || report.review_note }}</p>
    </template>

    <template v-else>
      <section class="workflow-section first-section" aria-label="管控措施库"><h4>管控措施库 <small>{{ control.measures.length }} 条</small></h4><details v-if="safety"><summary>从条文导入措施</summary><label class="workflow-label">条文引用编号<textarea v-model="measureReferences" rows="2" placeholder="每行一个已入库条文引用编号" aria-label="措施引用编号"></textarea></label><button class="workflow-button" :disabled="working || !parseReferenceIds(measureReferences).length" @click="act(async api => { await api.importMeasures(parseReferenceIds(measureReferences)); measureReferences = '' }, '措施已导入，等待审核')">导入措施</button></details><p v-if="!control.measures.length" class="workflow-muted">尚无可用措施。</p><article v-for="measure in control.measures" :key="identity(measure)" class="workflow-card"><div class="workflow-head"><strong>{{ measure.title || measure.name || '管控措施' }}</strong><span class="workflow-state">{{ stateLabel(measure) }}</span></div><p>{{ itemText(measure) }}</p><p class="workflow-muted">层级 {{ measure.tier || 1 }} · {{ roleNames[measure.assigned_role] || '待矿方确定' }} · {{ measure.deadline || '待矿方确定' }}</p><div class="workflow-citations"><button v-for="(reference,index) in refsFor(measure)" :key="reference.reference_id || index" class="workflow-link" @click="openCitation(reference)">{{ refLabel(reference) }}</button></div><template v-if="reviewAllowed(measure)"><label class="workflow-label">责任角色<select :value="measureRole(measure)" :aria-label="`措施责任角色 ${identity(measure)}`" @change="assignedRoleInputs[measure.measure_id] = $event.target.value"><option value="">待矿方确定</option><option v-for="(label,role) in roleNames" :key="role" :value="role">{{ label }}</option></select></label><label class="workflow-label">措施层级<select :value="tierInputs[measure.measure_id] ?? measure.tier ?? 1" :aria-label="`措施层级 ${identity(measure)}`" @change="tierInputs[measure.measure_id] = Number($event.target.value)"><option v-for="tier in [1,2,3,4,5]" :key="tier" :value="tier">第 {{ tier }} 级</option></select></label><label class="workflow-label">审核意见<textarea v-model="reviewNotes[notesKey('measure', measure)]" rows="2" :aria-label="`措施审核意见 ${identity(measure)}`"></textarea></label><div class="workflow-actions"><button class="workflow-button" :disabled="working || !measureRole(measure) || !reviewNotes[notesKey('measure', measure)]?.trim()" @click="review('measure', measure, 'approve')">通过措施</button><button class="workflow-button" :disabled="working || !reviewNotes[notesKey('measure', measure)]?.trim()" @click="review('measure', measure, 'reject')">退回措施</button></div></template></article></section>

      <section class="workflow-section" aria-label="事件管控计划"><div class="workflow-head"><h4>事件管控计划</h4><button class="workflow-button primary" :disabled="working || loading || workflowState(report) !== 'approved' || !reportCurrent" @click="createControlPlan">生成计划</button></div><p v-if="workflowState(report) !== 'approved'" class="workflow-muted">排查报告审核通过后可生成计划。</p><label v-if="control.plans.length" class="workflow-label">计划记录<select :value="selectedPlanId" @change="selectPlan"><option v-for="plan in control.plans" :key="plan.plan_id" :value="plan.plan_id">{{ plan.title || plan.plan_id }} · {{ stateLabel(plan) }}</option></select></label><template v-if="selectedPlan"><p class="workflow-muted">{{ stateLabel(selectedPlan) }} · 关联报告 {{ selectedPlan.report_id || '—' }}</p><p v-if="itemText(selectedPlan)">{{ itemText(selectedPlan) }}</p><p v-for="missing in selectedPlan.missing_fields || []" :key="missing" class="workflow-warning">{{ missingNames[missing] || '计划依据尚待补齐' }}</p><article v-for="item in selectedPlan.items || []" :key="item.measure_id" class="workflow-card"><p>{{ item.text }}</p><p class="workflow-muted">{{ roleNames[item.assigned_role] || '待分配' }} · 层级 {{ item.tier || 1 }} · 原文时限：{{ item.deadline || '待矿方确定' }}</p><button class="workflow-link" @click="openCitation(item)">查看措施原文</button><label v-if="reviewAllowed(selectedPlan)" class="workflow-label">任务完成时限（UTC+8）<input v-model="deadlineInputs[item.measure_id]" type="datetime-local" step="1" :aria-label="`任务完成时限 ${item.measure_id}`"></label></article><p v-if="selectedPlan.experience_ids?.length" class="workflow-muted">已引用经验：{{ selectedPlan.experience_ids.join('、') }}</p><template v-if="reviewAllowed(selectedPlan)"><label class="workflow-label">审核意见<textarea v-model="reviewNotes[notesKey('plan', selectedPlan)]" rows="2" aria-label="管控计划审核意见"></textarea></label><div class="workflow-actions"><button class="workflow-button primary" :disabled="working || !selectedPlan.items?.length || !reviewNotes[notesKey('plan', selectedPlan)]?.trim()" @click="review('plan', selectedPlan, 'approve')">批准计划</button><button class="workflow-button" :disabled="working || !reviewNotes[notesKey('plan', selectedPlan)]?.trim()" @click="review('plan', selectedPlan, 'reject')">退回计划</button></div></template></template></section>

      <section v-if="selectedPlan" class="workflow-section" aria-label="分角色执行任务"><h4>执行任务</h4><p v-if="!tasks.length" class="workflow-muted">当前计划尚无执行任务。</p><article v-for="task in tasks" :key="task.task_id" class="workflow-card"><div class="workflow-head"><strong>{{ task.title || task.name || '管控任务' }}</strong><span :class="['workflow-state', { overdue: isControlTaskOverdue(task, session.as_of) }]">{{ isControlTaskOverdue(task, session.as_of) ? '已逾期' : stateLabel(task) }}</span></div><p>{{ task.text || itemText(task) }}</p><dl class="workflow-facts"><div><dt>负责角色</dt><dd>{{ roleNames[task.responsible_role || task.assigned_role || task.role] || '待分配' }}</dd></div><div><dt>完成时限</dt><dd>{{ task.due_at ? mineTime(task.due_at) : task.deadline || '待矿方确定' }}</dd></div></dl><p v-if="task.execution?.content || task.execution_content || task.content" class="workflow-muted">执行记录：{{ task.execution?.content || task.execution_content || task.content }}</p><template v-if="['pending','assigned','new','ready'].includes(workflowState(task))"><label class="workflow-label">执行记录<textarea v-model="taskNotes[task.task_id]" rows="2" :aria-label="`任务执行记录 ${task.task_id}`"></textarea></label><button class="workflow-button" :disabled="!taskAllowed(task)" @click="act(async api => { await api.executeTask(task.task_id, taskNotes[task.task_id]); taskNotes[task.task_id] = '' }, '执行记录已保存')">提交执行记录</button></template></article></section>

      <section v-if="selectedPlan" class="workflow-section" aria-label="效果核验与复核"><div class="workflow-head"><h4>效果核验</h4><button class="workflow-button" :disabled="!canVerify" @click="act(api => api.verifyPlan(selectedPlan.plan_id), '核验结果已保存')">核验效果</button></div><p v-if="!effect" class="workflow-muted">任务执行完成后进行效果核验。</p><template v-else><dl class="workflow-facts"><div v-for="fact in effectFacts" :key="fact.label"><dt>{{ fact.label }}</dt><dd>{{ fact.value }}</dd></div></dl><div v-if="effect.before || effect.after" class="workflow-effect-table"><table><thead><tr><th>窗口</th><th>有效记录</th><th>平均值</th><th>最高值</th></tr></thead><tbody><tr v-for="window in [{key:'before',label:'处置前'},{key:'after',label:'处置后'}]" :key="window.key"><td>{{ window.label }}</td><td>{{ effect[window.key]?.count ?? '—' }}</td><td>{{ effect[window.key]?.mean ?? '—' }}</td><td>{{ effect[window.key]?.maximum ?? '—' }}</td></tr></tbody></table></div><p v-if="effect.policy" class="workflow-muted">核验阈值 {{ effect.policy.threshold ?? '待核对' }} {{ effect.policy.unit }} · {{ effect.policy.rule_status === 'team-draft' ? '团队草案' : '已登记规则' }}</p><p v-for="missing in effect.missing_fields || []" :key="missing" class="workflow-warning">{{ missingNames[missing] || '核验依据尚待补齐' }}</p><p v-if="itemText(effect)">{{ itemText(effect) }}</p><section v-if="effect.status === 'ineffective'" class="workflow-escalation"><h4>升级建议</h4><p class="workflow-warning">当前措施未达到核验目标，等待人工决定。</p><article v-for="measure in effect.escalation_candidates || []" :key="measure.measure_id" class="workflow-card"><p>{{ measure.text }}</p><small>措施层级 {{ measure.tier }}</small></article><p v-if="!effect.escalation_candidates?.length" class="workflow-muted">尚无已审核的更高层级适用措施。</p></section><p v-if="effect.escalation_required || effect.escalation_recommended" class="workflow-warning">建议升级处置{{ effect.escalation_reason ? `：${effect.escalation_reason}` : '' }}</p><ul v-if="effect.recommendations?.length"><li v-for="(item,index) in effect.recommendations" :key="index">{{ itemText(item) }}</li></ul><template v-if="safety && effect.status === 'effective' && !['completed','closed'].includes(workflowState(selectedPlan))"><label class="workflow-label">复核意见<textarea v-model="completionNote" rows="2" aria-label="管控完成复核意见"></textarea></label><button class="workflow-button primary" :disabled="working || !completionNote.trim()" @click="act(async api => { await api.reviewCompletion(selectedPlan.plan_id, completionNote); completionNote = '' }, '人工复核已保存')">复核完成</button></template></template></section>

      <section class="workflow-section" aria-label="管控经验"><h4>管控经验</h4><p v-if="!experiences.length && !planDetail?.experience" class="workflow-muted">复核完成后形成经验记录。</p><article v-for="experience in experiences.length ? experiences : planDetail?.experience ? [planDetail.experience] : []" :key="identity(experience)" class="workflow-card"><div class="workflow-head"><strong>{{ experience.title || '事件管控经验' }}</strong><span class="workflow-state">{{ stateLabel(experience) }}</span></div><p>{{ itemText(experience) || (experience.cause_ids || []).map(cause => causeNames[cause] || '待核实原因').join('、') }}</p><p v-if="experience.measure_ids?.length" class="workflow-muted">采用 {{ experience.measure_ids.length }} 项管控措施 · {{ experience.operation_context === 'rehearsal' ? '历史演练' : '现场记录' }}</p><div class="workflow-citations"><button v-for="(reference,index) in refsFor(experience)" :key="reference.reference_id || index" class="workflow-link" @click="openCitation(reference)">{{ refLabel(reference) }}</button></div><p v-if="experience.applicability || experience.conditions" class="workflow-muted">适用条件：{{ itemText(experience.applicability || experience.conditions) }}</p><p v-if="workflowState(experience) === 'approved'" class="workflow-muted">后续符合条件的事件可引用此经验。</p><template v-if="reviewAllowed(experience)"><label class="workflow-label">审核意见<textarea v-model="reviewNotes[notesKey('experience', experience)]" rows="2" :aria-label="`经验审核意见 ${identity(experience)}`"></textarea></label><div class="workflow-actions"><button class="workflow-button" :disabled="working || !reviewNotes[notesKey('experience', experience)]?.trim()" @click="review('experience', experience, 'approve')">批准经验</button><button class="workflow-button" :disabled="working || !reviewNotes[notesKey('experience', experience)]?.trim()" @click="review('experience', experience, 'reject')">退回经验</button></div></template></article></section>
    </template>

    <section v-if="citation" class="workflow-section workflow-citation-detail" aria-label="排查引用原文"><div class="workflow-head"><h4>{{ citation.title || citation.label || '引用依据' }}</h4><button class="workflow-button" @click="citation = null">关闭</button></div><p v-if="citationLoading" class="workflow-muted">正在读取原文…</p><p>{{ citation.text || citation.summary || citation.rationale || '该引用记录未附原文。' }}</p><small>{{ referenceLocation(citation.locator) }}</small><dl v-if="metricFacts(citation.metrics).length" class="workflow-facts"><div v-for="(fact,index) in metricFacts(citation.metrics)" :key="index"><dt>{{ fact.label }}</dt><dd>{{ fact.value }}</dd></div></dl><ul v-if="citation.evidence_refs?.length" class="workflow-source-refs"><li v-for="(reference,index) in citation.evidence_refs" :key="index"><span>{{ reference.observation_id || reference.run_id || reference.event_id || reference.reference_id || '事件记录' }}</span><small v-if="reference.event_time">{{ mineTime(reference.event_time) }}</small><small v-if="reference.source_file_id">来源 {{ reference.source_file_id }} · 原始行 {{ reference.row_number ?? '—' }}</small></li></ul><p class="workflow-muted">{{ citation.reference_id || citation.observation_id || citation.run_id || citation.event_id }}</p><p v-if="citation.event_time" class="workflow-muted">观测时间 {{ mineTime(citation.event_time) }}</p><p v-if="citation.value !== undefined" class="workflow-muted">记录值 {{ citation.value }}</p></section>
  </section>
</template>

<style scoped>
.workflow-label input{font-size:12px;padding:7px 8px;border:1px solid #34536c;background:#0c2336;color:#c4dbe8;border-radius:4px;margin-top:4px;min-width:0;width:100%;color-scheme:dark}.workflow-effect-table{overflow:auto;margin:10px 0}.workflow-effect-table table{border-collapse:collapse;width:100%;font-size:12px}.workflow-effect-table th,.workflow-effect-table td{padding:5px 7px;text-align:left;border-bottom:1px solid #29495f;white-space:nowrap;background:transparent;color:#bdd5e5}.workflow-source-refs{list-style:none;padding:0!important;display:grid;gap:7px}.workflow-source-refs li{display:grid;gap:2px}.workflow-escalation{margin-top:12px}
.event-workflow{font-size:13px;line-height:1.75;color:#c5d9e8;min-width:0;overflow-wrap:anywhere}.workflow-head{display:flex;justify-content:space-between;align-items:center;gap:8px}.workflow-head h3,.workflow-section h4{font-size:14px;font-weight:500;margin:0;color:#d4e6f2}.workflow-head strong{font-size:13px;font-weight:500}.workflow-section{margin-top:17px;padding-top:14px;border-top:1px solid #284156}.workflow-section.first-section{margin-top:0;padding-top:0;border-top:0}.workflow-section p,.workflow-section ul{margin-top:8px}.workflow-section ul{padding-left:19px}.workflow-muted{color:#91abc2;font-size:12px;margin:8px 0}.workflow-warning{color:#e6b978;font-size:12px;margin:8px 0}.workflow-notice{color:#8cd4c1;font-size:12px;margin:8px 0}.workflow-actions{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:10px}.workflow-button{border:1px solid #36586e;background:#142f43;color:#bbd9e9;border-radius:4px;font-size:12px;padding:4px 9px;line-height:1.6;white-space:nowrap}.workflow-button.primary{background:#16475a;color:#a5e5ed;border-color:#347387}.workflow-button:disabled{opacity:.45}.workflow-state{font-size:12px;color:#9bbcd0;border:1px solid #36566b;border-radius:4px;padding:0 6px;white-space:nowrap}.workflow-state.approved,.workflow-state.supported,.workflow-state.support{color:#8cd4c1;border-color:#33665b}.workflow-state.failed,.workflow-state.rejected,.workflow-state.opposed,.workflow-state.oppose,.workflow-state.overdue{color:#f2af9e;border-color:#7a4f45}.workflow-state.undetermined,.workflow-state.unknown{color:#e6b978;border-color:#65533b}.workflow-card{padding:11px;margin:10px 0;background:#102b40;border:1px solid #29495f;border-radius:5px}.workflow-label{font-size:12px;color:#91abc2;display:block;margin:9px 0}.workflow-label textarea,.workflow-label select{font-size:12px;padding:7px 8px;border-radius:4px;border:1px solid #34536c;background:#0c2336;color:#c4dbe8;min-width:0;width:100%;margin-top:4px}.workflow-label textarea{resize:vertical;min-height:56px}.workflow-steps{list-style:none;padding:0;margin:10px 0;display:grid;gap:8px}.workflow-steps li{border-left:2px solid #3c6b7f;padding:5px 0 5px 10px}.workflow-steps li>div{display:flex;justify-content:space-between;gap:8px;font-size:12px}.workflow-steps strong{font-weight:500}.workflow-steps p,.workflow-steps small{font-size:12px;color:#95afc3;margin:3px 0}.workflow-facts{display:grid;gap:7px;font-size:12px;margin:9px 0}.workflow-facts>div{display:grid;grid-template-columns:82px minmax(0,1fr);gap:8px}.workflow-facts dt{color:#91abc2}.workflow-facts dd{margin:0;font-variant-numeric:tabular-nums}.workflow-citations{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}.workflow-link{background:none;border:0;color:#7fd0df;font-size:12px;padding:1px 3px;text-decoration:underline;text-underline-offset:3px;text-align:left;overflow-wrap:anywhere}.workflow-citation-detail{padding:12px;background:#102b40;border:1px solid #375d72;border-radius:5px}.workflow-citation-detail>p{white-space:pre-wrap}.workflow-section summary{font-size:12px;color:#7fd0df;cursor:pointer;padding:6px 0}.workflow-section small{font-size:12px;color:#91abc2}@media(max-width:760px){.workflow-actions{gap:6px}.workflow-facts>div{grid-template-columns:75px minmax(0,1fr)}.workflow-card{padding:9px}}
</style>
