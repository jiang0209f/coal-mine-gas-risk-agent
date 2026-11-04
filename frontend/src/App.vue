<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { createApiClient } from './api.mjs'
import { advanceAsOf, chinaInputToIso, mineTime, observationWindow, queryString, seriesToPoints } from './utils.mjs'

echarts.use([LineChart, GridComponent, TooltipComponent, MarkLineComponent, CanvasRenderer])
const token = ref(''), user = ref(null), username = ref(''), password = ref('')
const busy = ref(''), error = ref(''), notice = ref(''), tab = ref('monitor')
const sources = ref([]), points = ref([]), sessions = ref([]), sourceId = ref(''), pointId = ref('')
const active = ref(null), mode = ref('historical_replay'), startTime = ref('2022-01-01T00:00')
const series = ref({ rows: [] }), events = ref([]), rules = ref(null), models = ref([]), reports = ref([])
const selectedEvent = ref(null), evidence = ref(null), caseResult = ref(null), selectedAdvice = ref('')
const caseContent = ref(''), refs = ref([]), referenceQuery = ref(''), searchResults = ref([]), selectedRefs = ref([])
const uploadFile = ref(null), importResult = ref(null), importStart = ref(''), importEnd = ref('')
const importLineage = ref('unknown')
const mappingText = ref('')
const documentForm = ref({ title: '', source: '', version: '1', text: '', page: 1 })
const chatInput = ref(''), messages = ref([]), chatBusy = ref(false), chatTask = ref(null)
const chartEl = ref(null)
let chart, observer, pollVersion = 0, playbackTimer
const roles = { ventilation: '通风人员', gas_governance: '瓦斯治理人员', safety_management: '安全管理人员' }
const levels = { red: '红色', orange: '橙色', yellow: '黄色', blue: '蓝色' }
const phases = { new: '待确认', confirmed: '已确认', advice_reviewed: '建议已审核', executed: '已执行', feedback_submitted: '待复核', reviewed: '已复核' }
const states = { running: '进行中', paused: '已暂停', ended: '已结束', completed: '已完成', failed: '失败', partial: '部分完成', awaiting_input: '待补充', cancelled: '已取消' }
const attributionNeeds = {
  observed_event_required: '等待实测预警事件', trained_model_for_source: '登记当前数据源的溯源模型',
  unambiguous_trained_model_version: '确定溯源模型版本', six_channel_point_bindings: '补齐六路测点映射',
  confirmed_processed_feature_points: '确认特征测点映射', target_point_binding: '补齐目标测点映射',
  observed_target_at_model_threshold: '等待目标测点达到溯源分析阈值', automatic_attribution_enabled: '开启事件溯源',
  aligned_six_channel_minute_history: '补齐六路同步分钟数据', valid_unique_minute_features: '补齐有效的分钟特征数据',
  complete_61_minute_six_channel_history: '补齐61个连续分钟时刻的六路数据', training_cutoff_not_after_as_of: '选择训练期后的监测时段',
  six_channel_feature_order: '确认六路特征顺序', complete_six_channel_history: '补齐完整六路历史数据',
  history_times: '补齐历史数据时间', continuous_60_second_history_ending_at_as_of: '补齐截止输入时刻的连续分钟数据',
  classifier: '补齐模型文件', training_metadata: '补齐模型登记信息', model_python: '配置溯源模型运行环境',
}
const qualityNames = { invalid: '无效值', missing: '缺测', unit_inferred: '单位推定', delayed: '延迟', suspected_calibration: '疑似标校', planned_calibration: '计划标校', confirmed_calibration: '已确认标校', duplicate_timestamp: '重复时间', timezone_inferred: '时区按配置', forward_filled: '过去值填补' }
const metricNames = { precision: '逐事件查准率', recall: '逐事件召回率', missed_events: '漏报事件数', false_alerts_per_valid_monitoring_hour: '每有效监测小时误报数', sample_precision: '逐样本查准率', sample_recall: '逐样本召回率', sample_fpr: '逐样本误报率', mae: '平均绝对误差', mape_with_zero_policy: '非零样本平均百分比误差', tool_task_completion_rate: '工具任务完成率', evidence_completeness: '依据完整率', advice_reference_coverage: '建议出处覆盖率' }
const safety = computed(() => user.value?.roles?.includes('safety_management'))
const source = computed(() => sources.value.find(item => item.source_id === sourceId.value))
const point = computed(() => points.value.find(item => item.point_id === pointId.value))
const chartPoints = computed(() => active.value ? seriesToPoints(series.value, active.value.as_of) : [])
const lastValue = computed(() => chartPoints.value.filter(item => item[1] !== null).at(-1)?.[1])
const redCount = computed(() => events.value.filter(item => item.alert_level === 'red').length)
const phase = computed(() => caseResult.value?.case?.phase || 'new')
const attribution = computed(() => evidence.value?.candidate_attribution)
const adviceOptions = computed(() => caseResult.value?.advice_versions || [])
const chosenAdvice = computed(() => adviceOptions.value.find(item => `${item.advice_id}:${item.version}` === selectedAdvice.value))
const selectedReview = computed(() => caseResult.value?.advice_review_status?.[selectedAdvice.value]?.decision)
const operationContext = computed(() => active.value?.mode === 'realtime' ? 'field_record' : 'rehearsal')
const api = createApiClient({ getToken: () => token.value, onUnauthorized: clearLogin })
const post = (path, body) => api(path, { method: 'POST', body })
const list = value => Array.isArray(value) ? value : []
const pretty = value => JSON.stringify(value, null, 2)
const qualityLabel = flag => qualityNames[flag] || flag

function clearLogin() {
  pollVersion++
  token.value = ''; user.value = null; active.value = null; chatBusy.value = false
  sources.value = []; sessions.value = []; selectedEvent.value = null; messages.value = []
}
async function run(label, callback) {
  busy.value = label; error.value = ''; notice.value = ''
  try { return await callback() } catch (caught) { error.value = caught.message } finally { busy.value = '' }
}
async function login() {
  await run('正在登录', async () => {
    const result = await post('/auth/login', { username: username.value, password: password.value })
    password.value = ''; token.value = result.access_token; user.value = result.user
    await loadBase()
  })
}
async function logout() {
  await run('退出登录', async () => { await post('/auth/logout', {}); clearLogin() })
}
async function loadBase() {
  const [sourceData, sessionData] = await Promise.all([api('/sources'), api('/sessions')])
  sources.value = list(sourceData); sessions.value = list(sessionData)
  sourceId.value = sources.value[0]?.source_id || ''
  await changeSource()
  await loadReferences()
  if (safety.value) reports.value = list(await api('/evaluations'))
}
async function changeSource() {
  points.value = sourceId.value ? list(await api(`/points?${queryString({ source_id: sourceId.value })}`)) : []
  pointId.value = points.value[0]?.point_id || ''
  if (active.value?.source_id !== sourceId.value) {
    active.value = null; series.value = { rows: [] }; events.value = []; rules.value = null; models.value = []
    selectedEvent.value = null; evidence.value = null; caseResult.value = null; pollVersion++; chatBusy.value = false
  }
  importLineage.value = 'unknown'
  resetMapping()
}
function resetMapping() {
  mappingText.value = pretty({ time_column: 'STime', value_columns: { SValue: pointId.value }, timezone: '+08:00', header_row: 1, lineage: 'unknown', invalid_values: [-999999] })
}
async function choosePoint() {
  resetMapping()
  selectedEvent.value = null; evidence.value = null; caseResult.value = null
  await run('加载点位', refreshWorkspace)
}
async function createSession() {
  await run('创建会话', async () => {
    active.value = await post('/sessions', { source_id: sourceId.value, mode: mode.value, start_as_of: chinaInputToIso(startTime.value) })
    sessions.value = list(await api('/sessions')); selectedEvent.value = null; messages.value = []; pollVersion++; chatBusy.value = false
    await refreshWorkspace()
  })
}
async function selectSession(event) {
  const id = event.target.value
  if (!id) return
  await run('加载会话', async () => {
    pollVersion++; chatBusy.value = false; selectedEvent.value = null; evidence.value = null; caseResult.value = null; messages.value = []
    active.value = await api(`/sessions/${encodeURIComponent(id)}`)
    sourceId.value = active.value.source_id
    await changeSource(); await refreshWorkspace()
  })
}
async function refreshWorkspace() {
  if (!active.value || !pointId.value) return
  const session = active.value
  const selectedPoint = pointId.value
  const params = { session_id: session.session_id, point_id: selectedPoint }
  const results = await Promise.allSettled([
    api(`/observations?${queryString({ ...params, ...observationWindow(session.as_of), aggregation_seconds: 60 })}`),
    api(`/events?${queryString(params)}`),
    api(`/rules?${queryString({ face_id: source.value?.face_id, point_id: pointId.value })}`),
    api(`/models?${queryString(params)}`),
  ])
  if (active.value?.session_id !== session.session_id || active.value?.as_of !== session.as_of || pointId.value !== selectedPoint) return
  const targets = [series, events, rules, models]
  results.forEach((result, index) => {
    if (result.status === 'fulfilled') targets[index].value = result.value
    else error.value = result.reason.message
  })
}
async function sessionAction(action) {
  await run('更新会话', async () => {
    await post(`/sessions/${active.value.session_id}/actions`, { action })
    active.value = await api(`/sessions/${active.value.session_id}`)
    await refreshWorkspace()
  })
}
async function tick(seconds = null) {
  if (!active.value || busy.value) return
  await run('推进数据时间', async () => {
    await post(`/sessions/${active.value.session_id}/tick`, seconds === null || active.value.mode === 'realtime' ? {} : { target_as_of: advanceAsOf(active.value.as_of, seconds) })
    active.value = await api(`/sessions/${active.value.session_id}`)
    await refreshWorkspace()
    if (selectedEvent.value) await loadEvent(selectedEvent.value)
  })
}
async function loadEvent(event) {
  selectedEvent.value = event
  const [bundle, caseData] = await Promise.all([
    api(`/evidence?${queryString({ session_id: active.value.session_id, event_id: event.event_id })}`),
    api(`/cases/${event.event_id}?${queryString({ session_id: active.value.session_id })}`),
  ])
  if (selectedEvent.value?.event_id !== event.event_id) return
  evidence.value = bundle; caseResult.value = caseData
  const latest = caseData.advice_versions?.at(-1)
  selectedAdvice.value = latest ? `${latest.advice_id}:${latest.version}` : ''
}
async function searchReferences() {
  await run('检索已批准资料', async () => {
    searchResults.value = list(await post('/references/search', { session_id: active.value.session_id, query: referenceQuery.value, event_id: selectedEvent.value?.event_id }))
    selectedRefs.value = []
  })
}
async function draftAdvice() {
  await run('生成带出处建议', async () => {
    await post('/advice', { session_id: active.value.session_id, event_id: selectedEvent.value.event_id, reference_ids: selectedRefs.value })
    await loadEvent(selectedEvent.value); notice.value = '已生成新的建议版本，请人工审核。'
  })
}
async function recordAction(actionType, decision) {
  await run('提交人工记录', async () => {
    await post(`/cases/${selectedEvent.value.event_id}/actions`, {
      session_id: active.value.session_id, action_type: actionType, operation_context: operationContext.value,
      content: caseContent.value, advice_id: chosenAdvice.value?.advice_id,
      advice_version: chosenAdvice.value?.version, review_decision: decision,
    })
    caseContent.value = ''; await loadEvent(selectedEvent.value); notice.value = '操作已保存，可在处置记录中回查。'
  })
}
async function importHistory() {
  await run('上传并导入数据', async () => {
    if (!uploadFile.value) throw new Error('请选择 CSV 或 XLSX 文件。')
    const mapping = JSON.parse(mappingText.value)
    mapping.lineage = importLineage.value
    const form = new FormData(); form.append('file', uploadFile.value)
    const file = await post('/files', form)
    importResult.value = await post('/history-imports', {
      source_id: sourceId.value, file_id: file.file_id, mapping,
      ...(importStart.value ? { start_at: chinaInputToIso(importStart.value) } : {}),
      ...(importEnd.value ? { end_at: chinaInputToIso(importEnd.value) } : {}),
    })
    notice.value = '导入完成，计数与质量结果见下方。'; await refreshWorkspace()
  })
}
async function loadReferences() { refs.value = list(await api('/references')) }
async function importDocument() {
  await run('登记资料', async () => {
    const form = documentForm.value
    await post('/references/imports', { spec: {
      title: form.title, source: form.source, version: form.version,
      applicability: { face_ids: [source.value?.face_id] },
      fragments: [{ text: form.text, locator: { page: Number(form.page) } }],
    } })
    documentForm.value.text = ''; await loadReferences(); notice.value = '资料已登记，批准后进入检索。'
  })
}
async function reviewDocument(doc, decision) {
  await run('审核资料', async () => {
    const base = doc.kind === 'experience' ? `/experiences/${doc.experience_id}` : `/references/${doc.document_id}`
    await post(`${base}/review`, { version: doc.version, decision, comment: '' })
    await loadReferences(); notice.value = '审核结论已保存。'
  })
}
async function sendChat() {
  const message = chatInput.value.trim()
  if (!message || !active.value || chatBusy.value) return
  chatInput.value = ''; messages.value.push({ role: 'user', text: message })
  chatBusy.value = true; chatTask.value = null; const version = pollVersion
  try {
    let task = await post('/chat', { session_id: active.value.session_id, message, async_execution: true })
    for (let count = 0; count < 90 && ['running', 'queued'].includes(task.state); count++) {
      if (version !== pollVersion) return
      chatTask.value = task
      await new Promise(resolve => setTimeout(resolve, 1000))
      if (version !== pollVersion) return
      task = await api(`/tasks/${task.task_id}`)
    }
    if (version !== pollVersion) return
    chatTask.value = task
    messages.value.push({ role: 'assistant', text: task.final_text || task.answer || task.message || `任务状态：${states[task.state] || task.state}`, state: task.state, task_id: task.task_id, method: task.brief_method, steps: task.tool_steps })
  } catch (caught) { if (version === pollVersion) messages.value.push({ role: 'assistant', text: caught.message, state: 'failed' }) }
  finally { if (version === pollVersion) chatBusy.value = false }
}
async function cancelChat() {
  if (!chatTask.value?.task_id) return
  await run('取消任务', async () => {
    chatTask.value = await post(`/tasks/${chatTask.value.task_id}/cancel`, {})
  })
}
function renderChart() {
  if (!chartEl.value) return
  if (chart && chart.getDom() !== chartEl.value) { observer?.disconnect(); chart.dispose(); chart = null }
  if (!chart) {
    chart = echarts.init(chartEl.value)
    observer = new ResizeObserver(() => chart?.resize()); observer.observe(chartEl.value)
  }
  const marks = rules.value ? [
    { name: '实测红色', yAxis: rules.value.observed_red_threshold, lineStyle: { color: '#d95c58' } },
    { name: '实测橙色', yAxis: rules.value.observed_orange_threshold, lineStyle: { color: '#d6a14c' } },
  ].filter(item => Number.isFinite(item.yAxis)) : []
  chart.setOption({ animation: false, textStyle: { fontFamily: 'Microsoft YaHei, sans-serif' },
    grid: { left: 48, right: 30, top: 24, bottom: 34 },
    tooltip: { trigger: 'axis', confine: true, formatter: entries => `${mineTime(entries[0]?.value?.[0])}<br>实测：${entries[0]?.value?.[1] ?? '缺测'} ${series.value.unit || ''}` },
    xAxis: { type: 'time', min: active.value ? Date.parse(observationWindow(active.value.as_of).start_at) : undefined,
      max: active.value ? Date.parse(active.value.as_of) : undefined,
      axisLine: { lineStyle: { color: '#dce3e8' } }, splitLine: { show: false }, axisLabel: { color: '#82909a', formatter: value => mineTime(value).slice(11, 16) } },
    yAxis: { type: 'value', min: 0, axisLabel: { color: '#82909a' }, splitLine: { lineStyle: { color: '#edf1f4' } } },
    series: [{ name: '实测浓度', type: 'line', data: chartPoints.value,
      showSymbol: chartPoints.value.filter(item => item[1] !== null).length < 2, symbolSize: 5, connectNulls: false,
      lineStyle: { width: 2, color: '#287b86' }, itemStyle: { color: '#287b86' },
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(40,123,134,0.13)' }, { offset: 1, color: 'rgba(40,123,134,0)' }]) },
      markLine: { silent: true, symbol: 'none', label: { formatter: '{b}', position: 'insideEndTop', fontSize: 10 }, lineStyle: { type: 'dashed', width: 1 }, data: marks },
    }],
  }, true)
}
watch([chartPoints, rules, tab, chartEl], () => nextTick(renderChart))
watch(active, () => {
  clearInterval(playbackTimer)
  if (active.value?.state === 'running') playbackTimer = setInterval(() => tick(), 5000)
})
onBeforeUnmount(() => { pollVersion++; clearInterval(playbackTimer); observer?.disconnect(); chart?.dispose() })
</script>

<template>
  <div v-if="!user" class="login-shell">
    <section class="login-intro"><div class="brand-icon">煤</div><p class="eyebrow">煤矿工作面安全风险管控</p><h1>从监测数据<br>到可回查的处置记录</h1><p>瓦斯监测 · 分级预警 · 依据回查 · 人工复核</p><div class="login-lines"><span></span><span></span><span></span></div><small>历史回放 / 模拟数据流 / 现场接入</small></section>
    <form class="login-card" @submit.prevent="login"><p class="eyebrow">工作台登录</p><h2>登录您的账户</h2><p class="muted">使用已配置的通风、瓦斯治理或安全管理账户。</p><label>用户名<input v-model="username" name="username" autocomplete="username" required placeholder="请输入用户名"></label><label>密码<input v-model="password" name="password" type="password" autocomplete="current-password" required placeholder="请输入密码"></label><p v-if="error" class="error" role="alert">{{ error }}</p><button class="primary full" :disabled="!!busy">{{ busy || '登录工作台' }} <span>→</span></button><small>登录后按账户角色和授权工作面显示功能。</small></form>
  </div>
  <div v-else class="app-shell">
    <aside class="sidebar"><div class="brand"><span class="brand-icon">煤</span><div>采面安全<small>瓦斯风险工作台</small></div></div><p class="nav-caption">工作空间</p><nav><button :class="{ active: tab === 'monitor' }" @click="tab = 'monitor'"><span>◫</span>监测与预警</button><button :class="{ active: tab === 'knowledge' }" @click="tab = 'knowledge'"><span>▤</span>资料与依据</button><button :class="{ active: tab === 'cases' }" @click="tab = 'cases'"><span>☑</span>处置与复核</button><button :class="{ active: tab === 'evaluation' }" @click="tab = 'evaluation'"><span>▥</span>模型与评估</button></nav><div class="sidebar-bottom"><span class="dot"></span>本地工作台<small>界面时间：矿区时间 UTC+8</small><button class="logout" @click="logout">退出登录</button></div></aside>
    <div class="workspace"><header class="topbar"><div><span class="breadcrumb">安全风险管控 / </span>{{ { monitor: '监测与预警', knowledge: '资料与依据', cases: '处置与复核', evaluation: '模型与评估' }[tab] }}</div><div class="user-chip"><span class="avatar">{{ (user.username || '用').slice(0, 1).toUpperCase() }}</span><div>{{ user.username || user.user_id }}<small>{{ user.roles.map(role => roles[role] || role).join(' / ') }}</small></div></div></header>
      <div class="page-body"><div class="page-heading"><div><p class="eyebrow">工作面监测与处置</p><h1>{{ source?.face_id ? `${source.face_id} 工作面` : '选择工作面数据源' }}</h1><p class="muted">{{ active ? '当前时点可见数据与事件，按来源逐项回查。' : '选择数据源并创建会话，开始查看监测数据。' }}</p></div><button class="secondary" :disabled="!active || !!busy" @click="run('刷新数据', refreshWorkspace)">↻ 刷新</button></div>
      <div v-if="error" class="notice error" role="alert">{{ error }}<button @click="error = ''" aria-label="关闭错误">×</button></div><div v-if="notice" class="notice success" role="status">{{ notice }}<button @click="notice = ''" aria-label="关闭通知">×</button></div><div v-if="busy" class="loading-note" role="status">{{ busy }}…</div>
      <section class="context-panel panel"><div class="context-selects"><label>数据源<select v-model="sourceId" @change="run('加载数据源', changeSource)"><option value="" disabled>请选择</option><option v-for="item in sources" :key="item.source_id" :value="item.source_id">{{ item.name || item.source_id }}</option></select></label><label>监测点<select v-model="pointId" @change="choosePoint"><option value="" disabled>请选择</option><option v-for="item in points" :key="item.point_id" :value="item.point_id">{{ item.name || item.source_tag || item.point_id }}</option></select></label><label>已有会话<select :value="active?.session_id || ''" @change="selectSession"><option value="">选择已有会话</option><option v-for="item in sessions" :key="item.session_id" :value="item.session_id">{{ mineTime(item.as_of, false) }} · {{ states[item.state] || item.state }}</option></select></label></div>
        <div v-if="active" class="session-strip"><span class="tag teal">{{ { historical_replay: '历史回放', simulated_stream: '模拟数据流', realtime: '现场接入' }[active.mode] }}</span><span class="session-clock">{{ mineTime(active.as_of) }}</span><span class="muted">{{ states[active.state] || active.state }}</span><div class="session-buttons"><button class="secondary small" :disabled="!!busy || active.state === 'ended'" @click="sessionAction(active.state === 'running' ? 'pause' : 'resume')">{{ active.state === 'running' ? 'Ⅱ 暂停' : '▷ 继续' }}</button><button class="primary small" :disabled="!!busy || active.state === 'ended'" @click="tick(60)">{{ active.mode === 'realtime' ? '刷新现场时间' : '前进 60 秒 →' }}</button></div></div>
        <details class="create-session" :open="!active"><summary>创建新会话</summary><div class="form-row"><label>运行方式<select v-model="mode"><option value="historical_replay">历史回放</option><option value="simulated_stream">模拟数据流</option><option value="realtime">现场接入</option></select></label><label>起始时间（UTC+8）<input v-model="startTime" type="datetime-local" step="1"></label><button class="primary" :disabled="!sourceId || !!busy" @click="createSession">创建会话</button></div></details>
      </section>
      <div class="content-grid"><main class="main-column">
      <template v-if="tab === 'monitor'">
        <div class="metrics-grid"><article class="metric-card"><span>最近有效实测值</span><strong>{{ lastValue === undefined ? '—' : lastValue.toFixed(3) }}<small>{{ series.unit || point?.unit || '%CH4' }}</small></strong><small>{{ point?.unit_status === 'inferred' ? '单位按资料推定' : point?.name || '尚未选择监测点' }}</small></article><article class="metric-card"><span>当前可见事件</span><strong>{{ active ? events.length : '—' }}<small>条</small></strong><small>其中红色 {{ redCount }} 条 · 实测与预测分别标记</small></article><article class="metric-card"><span>有效聚合记录</span><strong>{{ active ? series.quality?.valid_rows ?? chartPoints.filter(item => item[1] !== null).length : '—' }}<small>条</small></strong><small>最近 60 分钟 · 缺口保持可见</small></article></div>
        <section class="panel chart-panel"><div class="section-heading"><div><h2>瓦斯浓度变化</h2><p>实测 · 60 秒聚合 · {{ series.unit || point?.unit || '单位待核对' }}</p></div><span class="tag">{{ rules?.status === 'team-draft' ? '规则：团队草案' : rules?.rule_version || '规则待加载' }}</span></div><div ref="chartEl" class="chart" role="img" aria-label="当前会话时点之前的瓦斯实测浓度曲线"></div><div v-if="!chartPoints.length" class="chart-empty">{{ active ? '当前时间窗内无可见数据。可检查导入范围或推进会话时间。' : '创建会话后显示实测曲线。' }}</div><div class="chart-footer"><span><i class="line-key"></i>实测浓度</span><span>截止 {{ active ? mineTime(active.as_of) : '—' }}</span></div></section>
        <section class="panel"><div class="section-heading"><div><h2>预警事件</h2><p>点击事件查看触发依据和人工处置进度。</p></div><span class="count-badge">{{ events.length }}</span></div><div v-if="!events.length" class="empty-state"><span>◎</span><p>{{ active ? '当前时点没有可见事件' : '尚未创建会话' }}</p></div><div v-else class="event-list"><button v-for="event in [...events].sort((a,b) => Date.parse(b.start_at) - Date.parse(a.start_at))" :key="event.event_id" class="event-row" :class="{ selected: selectedEvent?.event_id === event.event_id }" @click="run('加载事件依据', () => loadEvent(event))"><span class="level-mark" :class="event.alert_level"></span><div><strong>{{ event.event_type === 'observed_exceedance' ? '实测超限' : '预测超限' }} <span class="tag" :class="event.alert_level">{{ levels[event.alert_level] || event.alert_level }}</span></strong><small>{{ mineTime(event.start_at) }} · {{ event.point_id }}</small><div class="flag-row"><span v-for="flag in event.quality_flags || []" :key="flag" class="tag">{{ qualityLabel(flag) }}</span></div></div><span class="event-arrow">查看 →</span></button></div></section>
        <section v-if="rules" class="panel rules-panel"><div class="section-heading"><div><h2>当前分级规则</h2><p>{{ rules.source }}</p></div><span class="tag amber">{{ rules.status === 'team-draft' ? '团队拟定 · 待矿方确认' : rules.rule_version }}</span></div><div class="rule-grid"><div><i class="level-mark red"></i><span>实测红色</span><strong>≥ {{ rules.observed_red_threshold }}%</strong></div><div><i class="level-mark orange"></i><span>实测橙色</span><strong>≥ {{ rules.observed_orange_threshold }}%<small>持续 {{ rules.orange_duration_seconds / 60 }} 分钟</small></strong></div><div><i class="level-mark yellow"></i><span>预测黄色</span><strong>≥ {{ rules.forecast_yellow_threshold }}%</strong></div></div></section>
        <section v-if="safety" class="panel"><details><summary>历史数据接入</summary><p class="muted">选择文件并核对字段映射；大表可按起止时间分段导入。</p><div class="form-row"><label>数据性质<select v-model="importLineage"><option value="unknown">来源待核实</option><option value="raw">原始监测记录</option><option value="processed">已处理数据</option></select></label><label>数据文件<input type="file" accept=".csv,.xlsx" @change="uploadFile = $event.target.files[0]"></label></div><div class="form-row"><label>开始时间（可选，UTC+8）<input v-model="importStart" type="datetime-local" step="1"></label><label>结束时间（可选，UTC+8）<input v-model="importEnd" type="datetime-local" step="1"></label></div><details><summary>高级配置：字段映射</summary><textarea v-model="mappingText" rows="10" class="code-input" aria-label="导入字段映射JSON"></textarea></details><button class="primary" :disabled="!!busy || !uploadFile || !sourceId" @click="importHistory">上传并导入</button><pre v-if="importResult" class="result-json">{{ pretty(importResult) }}</pre></details></section>
      </template>
      <template v-if="tab === 'knowledge'">
        <section class="panel"><div class="section-heading"><div><h2>已批准资料检索</h2><p>按当前工作面筛选，保留原文、来源与页段定位。</p></div></div><form class="search-form" @submit.prevent="searchReferences"><input v-model="referenceQuery" placeholder="输入条款关键词，如通风、甲烷、标校" aria-label="资料检索关键词"><button class="primary" :disabled="!active || !!busy">检索</button></form><p v-if="!active" class="muted">请先创建或选择会话。</p><article v-for="item in searchResults" :key="item.reference_id" class="reference-card"><div><span class="tag teal">{{ item.kind === 'experience' ? '已批准经验' : '已批准条文' }}</span><span class="tag" v-if="item.operation_context === 'rehearsal'">演练来源</span></div><p>{{ item.text }}</p><small>{{ item.source }} · {{ item.locator.page ? `第 ${item.locator.page} 页` : pretty(item.locator) }}</small></article><p v-if="active && !searchResults.length" class="muted">输入关键词检索；没有已批准的匹配资料时返回空结果。</p></section>
        <section class="panel"><div class="section-heading"><div><h2>资料版本</h2><p>批准后的版本才进入建议检索。</p></div><button class="secondary small" @click="run('刷新资料', loadReferences)">刷新</button></div><div v-if="!refs.length" class="empty-state"><p>暂无已登记资料</p></div><article v-for="doc in refs" :key="`${doc.entry_id}:${doc.version}`" class="document-row"><div><strong>{{ doc.title || (doc.kind === 'experience' ? '案例经验' : '资料') }}</strong><small>{{ doc.source }} · 版本 {{ doc.version }}</small></div><span class="tag" :class="doc.state === 'approved' ? 'teal' : 'amber'">{{ { approved: '已批准', rejected: '已拒绝', pending_review: '待审核', draft: '草稿' }[doc.state] || doc.state }}</span><div v-if="safety && doc.state === 'pending_review'" class="button-group"><button class="secondary small" @click="reviewDocument(doc, 'approved')">批准</button><button class="ghost small" @click="reviewDocument(doc, 'rejected')">拒绝</button></div></article></section>
        <section v-if="safety" class="panel"><details><summary>登记条文资料</summary><form @submit.prevent="importDocument"><div class="form-row"><label>资料名称<input v-model="documentForm.title" required></label><label>来源<input v-model="documentForm.source" required placeholder="填写真实资料来源"></label></div><div class="form-row"><label>版本<input v-model="documentForm.version" required></label><label>页码<input v-model="documentForm.page" type="number" min="1" required></label></div><label>条文原文<textarea v-model="documentForm.text" rows="5" required placeholder="录入待审核的真实原文"></textarea></label><button class="primary" :disabled="!source?.face_id || !!busy">登记并送审</button></form></details></section>
      </template>
      <template v-if="tab === 'cases' || (tab === 'monitor' && selectedEvent)">
        <section class="panel"><div class="section-heading"><div><h2>事件依据与处置</h2><p>{{ selectedEvent ? `${mineTime(selectedEvent.start_at)} · ${selectedEvent.point_id}` : '请在监测与预警中选择事件。' }}</p></div><span v-if="selectedEvent" class="tag teal">{{ operationContext === 'rehearsal' ? '演练记录' : '现场记录' }}</span></div>
          <template v-if="selectedEvent && evidence"><div class="evidence-grid"><div><span>实测依据片段</span><strong>{{ evidence.observation_fragments?.length || 0 }} 条</strong></div><div><span>事件修订</span><strong>第 {{ evidence.event_snapshot?.event_revision }} 版</strong></div><div><span>当前处置阶段</span><strong>{{ phases[phase] || phase }}</strong></div></div><p v-for="hint in evidence.condition_hints || []" :key="hint.kind" class="hint">{{ hint.text }}</p><p v-if="evidence.missing_fields?.length" class="hint amber-text">依据缺项：{{ evidence.missing_fields.map(item => ({ observation_fragments: '实测片段', forecast_run: '预测运行' }[item] || item)).join('、') }}</p><details><summary>查看原始依据片段</summary><div class="table-scroll"><table><thead><tr><th>观测时间</th><th>原始值</th><th>有效值</th><th>质量</th></tr></thead><tbody><tr v-for="row in evidence.observation_fragments || []" :key="row.id"><td>{{ mineTime(row.event_time) }}</td><td>{{ row.raw_value ?? '—' }}</td><td>{{ row.value ?? '缺测' }}</td><td>{{ row.quality_flags?.map(qualityLabel).join('、') || '有效' }}</td></tr></tbody></table></div></details>
          <section v-if="attribution" class="subsection attribution-card" aria-label="溯源结果">
            <div class="section-heading"><h3>溯源结果</h3><span class="tag" :class="attribution.status === 'completed' ? 'teal' : 'amber'">{{ { completed: '已完成', not_ready: '待补充', failed: '分析未完成' }[attribution.status] || '待分析' }}</span></div>
            <template v-if="attribution.status === 'completed'">
              <p class="hint"><strong>候选原因：</strong>{{ attribution.candidate_causes?.join('、') || '暂无候选原因' }}</p>
              <p class="muted">模型版本 {{ attribution.model_id }} · {{ attribution.model_version || attribution.version || '—' }}</p>
              <div class="evidence-grid"><div><span>目标时间</span><strong>{{ mineTime(attribution.target_time) }}</strong></div><div><span>输入截止</span><strong>{{ mineTime(attribution.input_as_of) }}</strong></div><div><span>模型提前量</span><strong>{{ attribution.horizon_seconds ?? '—' }} 秒</strong></div></div>
              <details v-if="attribution.cluster_probabilities?.length"><summary>类别概率</summary><div class="table-scroll"><table><thead><tr><th>类别</th><th>概率</th></tr></thead><tbody><tr v-for="item in attribution.cluster_probabilities" :key="item.cluster_id"><td>类别 {{ item.cluster_id }}</td><td>{{ (item.probability * 100).toFixed(1) }}%</td></tr></tbody></table></div></details>
            </template>
            <template v-else-if="attribution.status === 'not_ready'"><p v-for="need in attribution.missing_fields || []" :key="need" class="hint">{{ attributionNeeds[need] || '补齐溯源分析所需资料' }}</p><p v-if="!attribution.missing_fields?.length" class="hint">等待溯源分析所需数据。</p></template>
            <p v-else class="hint">{{ attribution.status === 'failed' ? '本次溯源分析未完成。' : '等待事件溯源分析。' }}</p>
          </section>
          <div class="subsection"><h3>引用资料并生成建议</h3><form class="search-form" @submit.prevent="searchReferences"><input v-model="referenceQuery" placeholder="检索已批准条文" aria-label="建议资料关键词"><button class="secondary" :disabled="!!busy">检索</button></form><label v-for="refItem in searchResults" :key="refItem.reference_id" class="reference-select"><input v-model="selectedRefs" type="checkbox" :value="refItem.reference_id"><span>{{ refItem.text }}<small>{{ refItem.source }} · {{ refItem.locator.page ? `第 ${refItem.locator.page} 页` : pretty(refItem.locator) }}</small></span></label><button class="primary small" :disabled="!selectedRefs.length || !!busy" @click="draftAdvice">生成带出处建议</button></div>
          <div class="subsection"><div class="section-heading"><h3>人工审核与执行</h3><span class="tag">{{ phases[phase] }}</span></div><label v-if="adviceOptions.length">建议版本<select v-model="selectedAdvice"><option v-for="item in adviceOptions" :key="`${item.advice_id}:${item.version}`" :value="`${item.advice_id}:${item.version}`">版本 {{ item.version }} · {{ { approved: '已批准', rejected: '已拒绝' }[caseResult.advice_review_status?.[`${item.advice_id}:${item.version}`]?.decision] || '待审核' }}</option></select></label><ol v-if="chosenAdvice" class="advice-items"><li v-for="item in chosenAdvice.items" :key="item.reference_id">{{ item.text }}<small>{{ item.source }} · {{ item.locator.page ? `第 ${item.locator.page} 页` : pretty(item.locator) }}</small></li></ol><label>操作说明<textarea v-model="caseContent" rows="3" placeholder="填写人工核实、执行、反馈或复核记录"></textarea></label><div class="action-buttons"><button class="secondary small" :disabled="phase !== 'new' || !caseContent.trim() || !!busy" @click="recordAction('confirm_event')">确认事件</button><template v-if="safety"><button class="secondary small" :disabled="!chosenAdvice || !['confirmed','advice_reviewed'].includes(phase) || !!busy" @click="recordAction('review_advice', 'approved')">批准建议</button><button class="ghost small" :disabled="!chosenAdvice || !['confirmed','advice_reviewed'].includes(phase) || !!busy" @click="recordAction('review_advice', 'rejected')">拒绝建议</button></template><button class="secondary small" :disabled="selectedReview !== 'approved' || !['confirmed','advice_reviewed'].includes(phase) || !caseContent.trim() || !!busy" @click="recordAction('record_execution')">记录执行</button><button class="secondary small" :disabled="phase !== 'executed' || !caseContent.trim() || !!busy" @click="recordAction('submit_feedback')">提交反馈</button><button class="primary small" :disabled="phase !== 'feedback_submitted' || !caseContent.trim() || !!busy" @click="recordAction('review_case')">完成复核</button></div></div>
          <div v-if="caseResult?.actions?.length" class="subsection"><h3>操作记录</h3><ol class="timeline"><li v-for="item in caseResult.actions" :key="item.action_id"><strong>{{ { confirm_event:'确认事件',review_advice:'审核建议',record_execution:'记录执行',submit_feedback:'提交反馈',review_case:'处置复核' }[item.action_type] }}</strong><p>{{ item.content || ({ approved:'批准',rejected:'拒绝' }[item.review_decision]) }}</p><small>{{ item.actor_id }} · 操作时间 {{ mineTime(item.occurred_at) }}<br>针对数据时点 {{ mineTime(item.data_as_of) }}</small></li></ol></div>
          </template><div v-else class="empty-state"><span>☑</span><p>选择事件后查看依据、审核建议和记录处置。</p><button class="secondary small" @click="tab = 'monitor'">前往监测与预警</button></div></section>
      </template>
      <template v-if="tab === 'evaluation'">
        <section class="panel"><div class="section-heading"><div><h2>已登记模型</h2><p>按当前会话数据源与监测点显示。</p></div></div><div v-if="!models.length" class="empty-state"><span>◈</span><p>{{ active ? '当前点位暂无已登记模型' : '选择会话后查看模型状态' }}</p></div><article v-for="model in models" :key="`${model.model_id}:${model.version}`" class="model-card"><div><strong>{{ model.name || model.model_id }}</strong><span class="tag" :class="model.model_type === 'baseline' ? 'amber' : 'teal'">{{ model.model_type === 'baseline' ? '离线基线' : '已登记专业模型' }}</span></div><p>{{ model.model_type === 'baseline' ? '用于离线流程验证。' : '是否可运行由输入完整性与适配器状态决定。' }}</p><small>版本 {{ model.version }} · 输入 {{ model.input_window }} 个时间桶 · 预测时长 {{ model.horizon_seconds }} 秒</small></article></section>
        <section class="panel"><div class="section-heading"><div><h2>冻结评估报告</h2><p>按实际报告展示指标、分子与分母。</p></div></div><div v-if="!safety" class="empty-state"><p>评估报告由安全管理人员查看。</p></div><div v-else-if="!reports.length" class="empty-state"><span>▥</span><p>暂无冻结评估报告</p><small>完成评估后显示实际结果。</small></div><article v-for="report in reports" :key="report.evaluation_id" class="report-card"><h3>{{ report.face_id }} 工作面 · {{ mineTime(report.generated_at, false) }}</h3><div class="table-scroll"><table><thead><tr><th>指标</th><th>结果</th><th>分子 / 分母</th></tr></thead><tbody><tr v-for="(value,key) in report.metric_values" :key="key"><td>{{ metricNames[key] || key }}</td><td>{{ value === null ? '分母为零，未计算' : typeof value === 'number' ? Number(value.toFixed(4)) : value }}</td><td>{{ report.numerators?.[key] ?? '—' }} / {{ report.denominators?.[key] ?? '—' }}</td></tr></tbody></table></div><details><summary>冻结清单与样本引用</summary><pre class="result-json">{{ pretty({ manifest_id: report.manifest_id, sample_refs: report.sample_refs }) }}</pre></details></article></section>
      </template>
      </main><aside class="assistant-panel panel"><div class="assistant-heading"><span class="assistant-symbol">✦</span><div><h2>风险管控助手</h2><small>调用实际工具 · 引用可查依据</small></div><span class="tag teal">DeepSeek</span></div><div class="assistant-context"><span class="dot"></span>{{ active ? `已绑定 ${source?.face_id || ''} 工作面会话` : '等待选择会话' }}</div><div class="chat-messages" aria-live="polite"><div v-if="!messages.length" class="assistant-welcome"><span>✦</span><h3>围绕当前数据时点提问</h3><p>可查询监测片段、预警事件、资料条文和处置进度。</p><button v-for="question in ['当前有哪些预警事件？','查看当前点位最近一小时的监测数据','这次事件有哪些可回查依据？']" :key="question" :disabled="!active" @click="chatInput = question">{{ question }} ↗</button></div><article v-for="(message,index) in messages" :key="index" class="chat-message" :class="message.role"><span class="message-label">{{ message.role === 'user' ? '我' : '风险管控助手' }}</span><p>{{ message.text }}</p><small v-if="message.state">{{ states[message.state] || message.state }}{{ message.method === 'structured_template' ? ' · 结构化模板' : '' }}</small><details v-if="message.steps?.length"><summary>工具调用记录</summary><pre>{{ pretty(message.steps) }}</pre></details></article><div v-if="chatBusy" class="chat-thinking"><span>{{ chatTask?.state ? (states[chatTask.state] || chatTask.state) : '正在提交任务' }}…</span><button v-if="chatTask?.task_id" class="ghost small" :disabled="!!busy || !['running','queued'].includes(chatTask.state)" @click="cancelChat">取消任务</button></div></div><form class="chat-form" @submit.prevent="sendChat"><textarea v-model="chatInput" rows="3" :disabled="!active" placeholder="向助手提问…" aria-label="向风险管控助手提问" @keydown.ctrl.enter.prevent="sendChat"></textarea><div><small>{{ active ? '围绕当前会话回答' : '请先创建会话' }}</small><button class="primary small" :disabled="!active || !chatInput.trim() || chatBusy">{{ chatBusy ? '处理中' : '发送 ↑' }}</button></div></form></aside></div>
      </div><footer class="page-footer">采煤工作面瓦斯超限预测及溯源管控智能体 <span>{{ active ? `${active.mode === 'realtime' ? '现场接入' : '演练回放'} · 数据时间 ${mineTime(active.as_of)}` : '尚未选择会话' }}</span></footer>
    </div>
  </div>
</template>
