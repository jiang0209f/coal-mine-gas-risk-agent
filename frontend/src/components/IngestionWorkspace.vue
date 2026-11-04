<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { mineTime, queryString } from '../utils.mjs'
import { createRequestGuard } from '../workspace.mjs'
import { batchStatusLabel, connectorStatusLabel, ingestionErrorLabel, ingestionSpec, workerStatusLabel } from '../ingestion.mjs'

const props = defineProps({ api: Function, safety: Boolean, sourceId: String, source: Object, points: Array, sessionId: String })
const emit = defineEmits(['notice', 'error'])
const connectors = ref([]), selectedId = ref(''), selected = ref(null), batches = ref(null), batchDetail = ref(null)
const worker = ref(null)
const loading = ref(false), working = ref(false), detailLoading = ref(false)
const listRequests = createRequestGuard(), detailRequests = createRequestGuard(), batchRequests = createRequestGuard()
const mutationRequests = createRequestGuard()
const scope = () => JSON.stringify([props.sourceId, props.sessionId])
const idOf = connector => connector?.connector_id || connector?.id || ''
const currentPoints = computed(() => (props.points || []).filter(point => point.source_id === props.sourceId))
const counts = computed(() => selected.value?.counts || {})
const form = ref({})
let pollTimer, mounted = true
function resetForm() {
  form.value = { connectorId: '', name: '', adapter: 'standard_json', staleSeconds: 120, rowsPath: '', idField: '', tagField: 'tag', timeField: 'time', valueField: 'value', sequenceField: '', timezone: props.source?.timezone || '+08:00', bindings: [{ tag: '', pointId: '' }] }
}
const time = (value, empty = '尚无记录') => value ? mineTime(value) : empty
const basePath = id => `/ingestion/connectors/${encodeURIComponent(id)}`
function clear() {
  clearTimeout(pollTimer); listRequests.invalidate(); detailRequests.invalidate(); batchRequests.invalidate(); mutationRequests.invalidate()
  connectors.value = []; worker.value = null; selectedId.value = ''; selected.value = null; batches.value = null; batchDetail.value = null
  loading.value = false; working.value = false; detailLoading.value = false
}
function schedulePoll() {
  clearTimeout(pollTimer)
  if (mounted && props.sourceId) pollTimer = setTimeout(async () => { if (!loading.value && !working.value) await refresh(); else schedulePoll() }, 5000)
}
watch(scope, () => { clear(); resetForm(); if (props.sourceId) void refresh() }, { immediate: true })
onBeforeUnmount(() => { mounted = false; clear() })
async function refresh() {
  if (!props.sourceId) return
  const requestedScope = scope(), request = listRequests.begin(); loading.value = true
  try {
    const result = await props.api(`/ingestion/status?${queryString({ source_id: props.sourceId })}`)
    if (!mounted || !listRequests.isCurrent(request) || requestedScope !== scope()) return
    connectors.value = result.connectors || []; worker.value = result.worker || null
    if (!connectors.value.some(item => idOf(item) === selectedId.value)) selectedId.value = idOf(connectors.value[0])
    if (selectedId.value) await loadSelected(batches.value?.offset || 0)
    else { detailRequests.invalidate(); selected.value = null; batches.value = null; batchDetail.value = null }
  } catch (error) { if (mounted && listRequests.isCurrent(request) && requestedScope === scope()) emit('error', error.message) }
  finally { if (mounted && listRequests.isCurrent(request) && requestedScope === scope()) { loading.value = false; schedulePoll() } }
}
async function loadSelected(offset = 0) {
  const id = selectedId.value
  if (!id) return
  const requestedScope = scope(), request = detailRequests.begin()
  try {
    const [detail, queue] = await Promise.all([props.api(basePath(id)), props.api(`${basePath(id)}/batches?${queryString({ offset, limit: 20 })}`)])
    if (!mounted || !detailRequests.isCurrent(request) || requestedScope !== scope() || selectedId.value !== id) return
    selected.value = detail; batches.value = queue
  } catch (error) { if (mounted && detailRequests.isCurrent(request) && requestedScope === scope() && selectedId.value === id) emit('error', error.message) }
}
function changeConnector() {
  detailRequests.invalidate(); batchRequests.invalidate(); selected.value = null; batches.value = null; batchDetail.value = null; detailLoading.value = false
  void loadSelected()
}
async function register() {
  if (!props.safety || working.value) return
  const requestedScope = scope(), request = mutationRequests.begin(); working.value = true
  try {
    const body = ingestionSpec(form.value, props.sourceId, currentPoints.value)
    const result = await props.api('/ingestion/connectors', { method: 'POST', body })
    if (!mounted || !mutationRequests.isCurrent(request) || requestedScope !== scope()) return
    selectedId.value = idOf(result); resetForm(); await refresh(); emit('notice', '接入已登记，启用后等待采集端发送数据。')
  } catch (error) { if (mounted && mutationRequests.isCurrent(request) && requestedScope === scope()) emit('error', error.message) }
  finally { if (mounted && mutationRequests.isCurrent(request) && requestedScope === scope()) working.value = false }
}
async function act(action) {
  if (!props.safety || working.value || !selectedId.value) return
  const id = selectedId.value, requestedScope = scope(), request = mutationRequests.begin(); working.value = true
  try {
    await props.api(`${basePath(id)}/actions`, { method: 'POST', body: { action } })
    if (!mounted || !mutationRequests.isCurrent(request) || requestedScope !== scope() || selectedId.value !== id) return
    batchDetail.value = null; await refresh()
  } catch (error) { if (mounted && mutationRequests.isCurrent(request) && requestedScope === scope() && selectedId.value === id) emit('error', error.message) }
  finally { if (mounted && mutationRequests.isCurrent(request) && requestedScope === scope()) working.value = false }
}
async function openBatch(batch) {
  const id = selectedId.value, requestedScope = scope(), request = batchRequests.begin()
  batchDetail.value = null; detailLoading.value = true
  try {
    const result = await props.api(`${basePath(id)}/batches/${encodeURIComponent(batch.batch_id)}`)
    if (!mounted || !batchRequests.isCurrent(request) || requestedScope !== scope() || selectedId.value !== id) return
    batchDetail.value = result
  } catch (error) { if (mounted && batchRequests.isCurrent(request) && requestedScope === scope()) emit('error', error.message) }
  finally { if (mounted && batchRequests.isCurrent(request)) detailLoading.value = false }
}
</script>

<template>
  <div class="ingestion-workspace">
    <section class="panel ingestion-overview">
      <div class="section-heading"><div><h2>现场数据接入</h2><p>来源 {{ source?.name || sourceId || '待选择' }} · {{ currentPoints.length }}个已登记测点</p></div><button class="secondary small" :disabled="loading || !sourceId" @click="refresh">刷新接入状态</button></div>
      <p class="muted">接收监测批次，核对测点与时间，保存入库结果及补传记录。</p>
      <div class="ingestion-worker" :class="{ warn: worker?.state === 'degraded' }" role="status" aria-label="补传服务状态"><strong>补传服务：{{ workerStatusLabel(worker) }}</strong><span>最近扫描 {{ time(worker?.last_run_at) }}</span><span>最近成功扫描 {{ time(worker?.last_success_at) }}</span><span v-if="worker?.last_error_code">后台处理异常，待处理批次已保留。</span></div>
      <label v-if="connectors.length" class="connector-select">接入配置<select v-model="selectedId" @change="changeConnector"><option v-for="item in connectors" :key="idOf(item)" :value="idOf(item)">{{ item.name }}</option></select></label>
      <p v-else class="empty-state">{{ loading ? '正在读取接入配置。' : '当前来源尚未登记现场接入。' }}</p>
    </section>
    <section v-if="selected" class="panel" role="region" aria-label="接入运行状态">
      <div class="section-heading"><div><h2>{{ selected.name }}</h2><p>{{ selected.adapter_kind === 'mapped_json' ? '字段映射接入' : '标准格式接入' }} · 当前来源</p></div><span class="ingestion-status" :class="{ warn: ['stale','retry_wait','error'].includes(selected.status) }">{{ connectorStatusLabel(selected) }}</span></div>
      <div v-if="safety" class="ingestion-actions"><button v-if="!selected.enabled" class="primary small" :disabled="working" @click="act('enable')">启用接入</button><button v-else class="secondary small" :disabled="working" @click="act('pause')">暂停接入</button><button class="secondary small" :disabled="working || !selected.enabled || !(counts.retry_batches || counts.rejected_batches)" @click="act('retry')">立即重试</button></div>
      <div class="ingestion-metrics"><div><span>待处理批次</span><strong>{{ (counts.pending_batches || 0) + (counts.retry_batches || 0) }}</strong><small>{{ counts.queued_rows || 0 }}条待处理记录</small></div><div><span>已入库记录</span><strong>{{ counts.ingested_rows || 0 }}</strong><small>{{ counts.completed_batches || 0 }}个已完成批次</small></div><div><span>去重记录</span><strong>{{ counts.duplicate_rows || 0 }}</strong><small>{{ counts.duplicate_batches || 0 }}次重复批次提交</small></div><div><span>等待重试</span><strong>{{ counts.retry_batches || 0 }}</strong><small>{{ counts.rejected_batches || 0 }}个待核查批次</small></div></div>
      <dl class="ingestion-facts"><div><dt>最近接收</dt><dd>{{ time(selected.last_received_at, '尚未收到批次') }}</dd></div><div><dt>最近成功入库</dt><dd>{{ time(selected.last_success_at, '尚无成功入库') }}</dd></div><div><dt>最新监测时间</dt><dd>{{ time(selected.latest_event_time, '尚无已接收监测记录') }}</dd></div><div><dt>最近采集端心跳</dt><dd>{{ time(selected.last_heartbeat_at, '尚无心跳记录') }}</dd></div></dl>
      <p v-if="selected.last_error_code" class="hint amber-text">{{ ingestionErrorLabel(selected.last_error_code) }}</p>
      <p v-if="counts.conflict_batches" class="hint amber-text">{{ counts.conflict_batches }}个批次存在编号或内容冲突，请核对采集端记录。</p>
      <details class="ingestion-settings"><summary>测点与字段配置</summary><p>{{ selected.adapter_kind === 'mapped_json' ? '按上游标签映射到本来源已登记测点。' : '按标准记录中的测点编号匹配本来源已登记测点。' }}</p><ul v-if="Object.keys(selected.tag_map || {}).length" class="binding-list"><li v-for="(pointId,tag) in selected.tag_map" :key="tag"><span>{{ tag }}</span><span>{{ currentPoints.find(point => point.point_id === pointId)?.name || pointId }}</span></li></ul><dl v-if="selected.adapter_kind === 'mapped_json'" class="ingestion-facts"><div><dt>记录路径</dt><dd>{{ selected.mapping?.records_path || '根数组' }}</dd></div><div><dt>标签字段</dt><dd>{{ selected.mapping?.tag_field }}</dd></div><div><dt>时间字段</dt><dd>{{ selected.mapping?.time_field }}</dd></div><div><dt>数值字段</dt><dd>{{ selected.mapping?.value_field }}</dd></div><div><dt>原始时间时区</dt><dd>{{ selected.mapping?.timezone || '来源时区' }}</dd></div></dl></details>
    </section>
    <section v-if="selected && batches" class="panel" role="region" aria-label="服务端批次队列">
      <div class="section-heading"><div><h2>批次与补传</h2><p>显示服务端已持久接收的批次。采集端尚未送达的记录由采集端管理。</p></div><span class="muted">共{{ batches.total }}个批次</span></div>
      <div class="table-scroll"><table class="ingestion-table"><thead><tr><th>批次</th><th>处理状态</th><th>记录数</th><th>接收时间</th><th>处理次数</th><th>处理结果</th></tr></thead><tbody><tr v-for="batch in batches.items || []" :key="batch.batch_id"><td><button class="ghost small batch-link" :aria-label="`查看批次 ${batch.batch_id}`" @click="openBatch(batch)">{{ batch.batch_id }}</button></td><td>{{ batchStatusLabel(batch.status) }}</td><td>{{ batch.row_count ?? 0 }}</td><td>{{ time(batch.accepted_at) }}</td><td>{{ batch.attempt_count || 0 }}</td><td>{{ batch.status === 'completed' ? `新增${batch.imported_rows || 0}条 · 重复${batch.duplicate_rows || 0}条` : ingestionErrorLabel(batch.error_code) || '等待处理' }}<small v-if="batch.next_retry_at">下次 {{ time(batch.next_retry_at) }}</small></td></tr></tbody></table></div>
      <p v-if="!batches.items?.length" class="muted">尚无服务端接收批次。</p>
      <div v-if="batches.total > batches.limit" class="ingestion-pagination"><button class="secondary small" :disabled="loading || !batches.offset" @click="loadSelected(Math.max(0,batches.offset - batches.limit))">上一页</button><button class="secondary small" :disabled="loading || batches.offset + batches.limit >= batches.total" @click="loadSelected(batches.offset + batches.limit)">下一页</button></div>
      <p v-if="detailLoading" class="muted">正在读取批次摘要。</p>
      <article v-if="batchDetail" class="batch-detail" role="region" aria-label="批次摘要"><div class="section-heading"><h3>{{ batchDetail.batch_id }}</h3><button class="ghost small" @click="batchDetail = null; batchRequests.invalidate()">关闭摘要</button></div><p>{{ batchStatusLabel(batchDetail.status) }} · 新增{{ batchDetail.imported_rows || 0 }}条 · 重复{{ batchDetail.duplicate_rows || 0 }}条</p><dl class="ingestion-facts"><div><dt>接收时间</dt><dd>{{ time(batchDetail.accepted_at) }}</dd></div><div><dt>成功入库</dt><dd>{{ time(batchDetail.applied_at, '尚未入库') }}</dd></div><div><dt>处理次数</dt><dd>{{ batchDetail.attempt_count || 0 }}</dd></div><div><dt>下次重试</dt><dd>{{ time(batchDetail.next_retry_at, '无待执行重试') }}</dd></div></dl><p v-if="batchDetail.error_code" class="amber-text">{{ ingestionErrorLabel(batchDetail.error_code) }}</p></article>
    </section>
    <section v-if="safety && sourceId" class="panel">
      <div class="section-heading"><div><h2>登记接入</h2><p>选择本来源的监测格式与测点映射，登记后由安全管理人员启用。</p></div></div>
      <form class="ingestion-form" @submit.prevent="register"><div class="ingestion-form-grid"><label>接入名称<input v-model="form.name" required maxlength="120"></label><label>接入编号（可选）<input v-model="form.connectorId" maxlength="128" placeholder="留空自动生成"></label><label>数据格式<select v-model="form.adapter" aria-label="数据格式"><option value="standard_json">标准监测记录</option><option value="mapped_json">自定义字段映射</option></select></label><label>数据中断判定（秒）<input v-model="form.staleSeconds" type="number" min="1" step="1" required></label></div>
        <p v-if="form.adapter === 'standard_json'" class="muted">标准记录包含测点编号、采集时间和原始监测值，按当前来源的测点信息核对。</p>
        <template v-else><div class="ingestion-form-grid"><label>记录路径<input v-model="form.rowsPath" placeholder="留空使用根数组"></label><label>记录编号字段（可选）<input v-model="form.idField"></label><label>标签字段<input v-model="form.tagField" required></label><label>时间字段<input v-model="form.timeField" required></label><label>数值字段<input v-model="form.valueField" required></label><label>序号字段（可选）<input v-model="form.sequenceField"></label><label>原始时间时区<input v-model="form.timezone" placeholder="+08:00"></label></div><p class="muted">记录路径和字段支持逐级名称，如 data.records。时区用于原始记录中未带时区的时间。</p><h3>上游标签与测点</h3><div v-for="(binding,index) in form.bindings" :key="index" class="binding-row"><label>上游测点标签 {{ index + 1 }}<input v-model="binding.tag" required></label><label>对应测点 {{ index + 1 }}<select v-model="binding.pointId" :aria-label="`对应测点 ${index + 1}`" required><option value="">选择已登记测点</option><option v-for="point in currentPoints" :key="point.point_id" :value="point.point_id">{{ point.name || point.point_id }}</option></select></label><button type="button" class="ghost small" :disabled="form.bindings.length < 2" :aria-label="`移除映射 ${index + 1}`" @click="form.bindings.splice(index,1)">移除</button></div><button type="button" class="secondary small" @click="form.bindings.push({tag:'',pointId:''})">添加测点映射</button></template>
        <div class="ingestion-submit"><button class="primary" :disabled="working || !currentPoints.length">{{ working ? '正在保存…' : '登记接入' }}</button><span class="muted">{{ currentPoints.length ? '登记状态：未启用' : '请先登记当前来源的测点。' }}</span></div>
      </form>
    </section>
  </div>
</template>

<style scoped>
.ingestion-workspace{display:grid;gap:18px;min-width:0}.ingestion-workspace .panel{min-width:0}.ingestion-workspace .section-heading>div{min-width:0;max-width:100%}.ingestion-workspace .section-heading h2,.ingestion-workspace .section-heading p{overflow-wrap:anywhere}.ingestion-overview .muted{margin:0 0 16px}.ingestion-worker{display:flex;flex-wrap:wrap;gap:7px 18px;align-items:center;margin-bottom:18px;padding:10px 12px;border:1px solid #28465f;border-radius:7px;background:#10283e;font-size:12px;color:#91a9bd}.ingestion-worker strong{font-weight:500;color:#9dd8ca}.ingestion-worker.warn strong,.ingestion-worker.warn>span:last-child{color:#efc77a}.connector-select{max-width:460px;display:grid;gap:7px}.ingestion-status{font-size:13px;padding:6px 12px;border:1px solid #2b6272;border-radius:20px;background:#123d46;color:#96e1d0;white-space:nowrap}.ingestion-status.warn{background:#44331e;border-color:#72552b;color:#efc77a}.ingestion-actions{display:flex;gap:10px;margin:14px 0 20px}.ingestion-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0 22px}.ingestion-metrics>div{background:#0e2439;border:1px solid #213c54;border-radius:9px;padding:15px 18px;display:grid;gap:6px}.ingestion-metrics span,.ingestion-metrics small{color:#96aabd;font-size:12px}.ingestion-metrics strong{font-size:28px;color:#dbe9f3;font-weight:600}.ingestion-facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px 24px;margin:15px 0}.ingestion-facts div{min-width:0}.ingestion-facts dt{color:#89a0b5;font-size:12px;margin-bottom:5px}.ingestion-facts dd{margin:0;overflow-wrap:anywhere;color:#d5e3ef;font-size:13px}.ingestion-settings{margin-top:18px;border-top:1px solid #27415a;padding-top:16px}.ingestion-settings summary{color:#aec7dc;cursor:pointer}.binding-list{padding:0;list-style:none}.binding-list li{display:flex;flex-wrap:wrap;gap:16px;padding:8px 0;border-bottom:1px solid #223c53;overflow-wrap:anywhere}.ingestion-table{width:100%;min-width:750px;border-collapse:collapse;font-size:13px}.ingestion-table th,.ingestion-table td{text-align:left;border-bottom:1px solid #223c53;padding:12px 10px;vertical-align:top}.ingestion-table th{color:#8ea8bd;font-weight:500}.ingestion-table small{display:block;color:#94aabd;margin-top:5px}.batch-link{max-width:210px;white-space:normal;text-align:left;overflow-wrap:anywhere}.ingestion-pagination{display:flex;justify-content:flex-end;gap:10px;margin-top:15px}.batch-detail{margin-top:18px;padding:16px;background:#0e2439;border:1px solid #28465f;border-radius:8px;overflow-wrap:anywhere}.batch-detail h3{margin:0}.batch-detail .section-heading{margin:0}.ingestion-form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px 20px;margin-bottom:16px}.ingestion-form label,.binding-row label{display:grid;gap:7px;min-width:0;font-size:13px;color:#b5c9da}.ingestion-form input,.ingestion-form select,.connector-select select{width:100%;min-width:0}.ingestion-form h3{font-size:14px;margin:22px 0 12px}.binding-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) auto;gap:12px;align-items:end;margin-bottom:12px}.ingestion-submit{margin-top:24px;display:flex;flex-wrap:wrap;gap:14px;align-items:center}.ingestion-form .muted,.ingestion-workspace .section-heading p{font-size:12px;line-height:1.7}.ingestion-workspace .section-heading{gap:12px;flex-wrap:wrap}.table-scroll{max-width:100%;overflow-x:auto}
@media(max-width:900px){.ingestion-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:600px){.ingestion-form-grid,.ingestion-facts{grid-template-columns:minmax(0,1fr)}.ingestion-metrics{gap:8px}.ingestion-metrics>div{padding:12px}.ingestion-metrics strong{font-size:24px}.binding-row{grid-template-columns:minmax(0,1fr)}.binding-row button{justify-self:start}.ingestion-workspace .section-heading{align-items:flex-start}.batch-detail{padding:12px}.ingestion-submit{align-items:flex-start;flex-direction:column}}
</style>
