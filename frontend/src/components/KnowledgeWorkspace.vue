<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { createRequestGuard } from '../workspace.mjs'
import { referenceLocation, referenceValidity } from '../knowledge.mjs'
import { chinaInputToIso, mineTime } from '../utils.mjs'

const props = defineProps({ api: Function, entries: Array, faceId: String, sourceId: String, session: Object, safety: Boolean })
const emit = defineEmits(['refresh', 'notice', 'error'])
const detail = ref(null), fileMetadata = ref(null), selectedFile = ref(null), mode = ref('file'), loading = ref(false), working = ref(false)
const updatingEntry = ref(null)
const search = ref(''), results = ref([]), searched = ref(false), stateFilter = ref(''), reviewComment = ref('')
const fileSelection = ref(0)
const form = ref({ title: '', source: '', version: '1', text: '', page: 1, validFrom: '', validTo: '' })
const detailRequests = createRequestGuard(), searchRequests = createRequestGuard()
const scope = () => JSON.stringify([props.faceId, props.sourceId, props.session?.session_id, props.session?.source_id])
const entries = computed(() => (props.entries || []).filter(item => (!item.applicability?.face_ids || item.applicability.face_ids.includes(props.faceId)) && (!stateFilter.value || item.state === stateFilter.value)))
const names = { approved: '已批准', rejected: '已拒绝', pending_review: '待审核', draft: '草稿', superseded: '已替代' }
const effectiveNames = { expired: '已过期', not_effective: '未生效' }

watch(scope, () => { detailRequests.invalidate(); searchRequests.invalidate(); detail.value = null; fileMetadata.value = null; loading.value = false; results.value = []; searched.value = false; reviewComment.value = ''; resetForm() })
onBeforeUnmount(() => { detailRequests.invalidate(); searchRequests.invalidate() })
async function act(callback) {
  working.value = true
  try { await callback() } catch (error) { emit('error', error.message) } finally { working.value = false }
}
function selectFile(event) {
  selectedFile.value = event.target.files?.[0] || null
  if (selectedFile.value && !form.value.title) form.value.title = selectedFile.value.name.replace(/\.[^.]+$/, '')
}
function resetForm() {
  updatingEntry.value = null; selectedFile.value = null; fileSelection.value++
  form.value = { title: '', source: '', version: '1', text: '', page: 1, validFrom: '', validTo: '' }
}
function updateDocument(entry) {
  resetForm(); updatingEntry.value = entry; mode.value = 'file'
  form.value = { ...form.value, title: entry.title, source: entry.source, version: '',
    validFrom: entry.applicability?.valid_from ? mineTime(entry.applicability.valid_from).replace(' ', 'T').slice(0, 16) : '',
    validTo: entry.applicability?.valid_to ? mineTime(entry.applicability.valid_to).replace(' ', 'T').slice(0, 16) : '' }
}
async function importDocument() {
  await act(async () => {
    const requestedScope = scope(), faceId = props.faceId, spec = { title: form.value.title, source: form.value.source, version: form.value.version, applicability: { face_ids: [faceId] } }
    if (updatingEntry.value) spec.entry_id = updatingEntry.value.entry_id
    if (form.value.validFrom) spec.applicability.valid_from = chinaInputToIso(form.value.validFrom)
    if (form.value.validTo) spec.applicability.valid_to = chinaInputToIso(form.value.validTo)
    if (mode.value === 'file') {
      const file = selectedFile.value
      if (!file) throw new Error('请选择资料文件。')
      if (!/\.(pdf|docx|txt|md)$/i.test(file.name)) throw new Error('资料文件支持 PDF、DOCX、TXT 和 MD。')
      if (file.size > 20 * 1024 * 1024) throw new Error('资料文件不能超过20 MiB。')
      const upload = new FormData(); upload.append('file', file); upload.append('face_ids', JSON.stringify([faceId]))
      const metadata = await props.api('/references/files', { method: 'POST', body: upload })
      if (scope() !== requestedScope) return
      spec.file_id = metadata.file_id
    } else spec.fragments = [{ text: form.value.text, locator: { page: Number(form.value.page) } }]
    const result = await props.api('/references/imports', { method: 'POST', body: { spec } })
    if (scope() !== requestedScope) return
    form.value.text = ''; selectedFile.value = null; updatingEntry.value = null; fileSelection.value++
    emit('refresh'); emit('notice', '资料已送审，批准后进入检索。')
    await openDocument(result)
  })
}
async function openDocument(entry) {
  const request = detailRequests.begin(), requestedScope = scope()
  loading.value = true; detail.value = null; fileMetadata.value = null; reviewComment.value = ''
  try {
    const document = await props.api(`/references/${encodeURIComponent(entry.entry_id || entry.document_id || entry.experience_id)}?version=${encodeURIComponent(entry.version)}`)
    let metadata = null
    if (document.file_id) metadata = await props.api(`/references/files/${encodeURIComponent(document.file_id)}`)
    if (detailRequests.isCurrent(request) && requestedScope === scope()) { detail.value = document; fileMetadata.value = metadata }
  } catch (error) { if (detailRequests.isCurrent(request)) emit('error', error.message) }
  finally { if (detailRequests.isCurrent(request)) loading.value = false }
}
async function review(decision) {
  await act(async () => {
    const document = detail.value, requestedScope = scope()
    const base = document.kind === 'experience' ? `/experiences/${document.experience_id || document.entry_id}` : `/references/${document.document_id || document.entry_id}`
    await props.api(`${base}/review`, { method: 'POST', body: { version: document.version, decision, comment: reviewComment.value } })
    if (requestedScope !== scope()) return
    emit('refresh')
    if (detail.value?.entry_id !== document.entry_id || detail.value?.version !== document.version) return
    await openDocument(document); emit('notice', '审核结论已保存。')
  })
}
async function find() {
  const request = searchRequests.begin(), requestedScope = scope(), session = props.session
  if (!session) return
  await act(async () => {
    const found = await props.api('/references/search', { method: 'POST', body: { session_id: session.session_id, query: search.value } })
    if (searchRequests.isCurrent(request) && requestedScope === scope()) { results.value = found; searched.value = true }
  })
}
async function download() {
  await act(async () => {
    const file = fileMetadata.value
    const blob = await props.api(`/references/files/${encodeURIComponent(file.file_id)}/content`, { responseType: 'blob' })
    const url = URL.createObjectURL(blob), anchor = document.createElement('a')
    anchor.href = url; anchor.download = file.filename; anchor.click(); URL.revokeObjectURL(url)
  })
}
function close() { detailRequests.invalidate(); detail.value = null; fileMetadata.value = null; loading.value = false }
</script>

<template>
  <div class="knowledge-workspace">
    <section class="panel">
      <div class="section-heading"><div><h2>资料库</h2><p>按工作面查阅资料版本、审核状态和原文。</p></div><button class="secondary small" :disabled="working" @click="emit('refresh')">刷新资料</button></div>
      <label class="state-filter">审核状态<select v-model="stateFilter"><option value="">全部状态</option><option value="pending_review">待审核</option><option value="approved">已批准</option><option value="rejected">已拒绝</option><option value="superseded">已替代</option></select></label>
      <article v-for="entry in entries" :key="`${entry.entry_id}:${entry.version}`" class="document-row">
        <div class="document-name"><strong>{{ entry.title || (entry.kind === 'experience' ? '处置经验' : '资料') }}</strong><small>{{ entry.source }} · 版本 {{ entry.version }}<span v-if="entry.fragment_count"> · {{ entry.fragment_count }}个片段</span></small></div>
        <span v-if="entry.operation_context === 'rehearsal'" class="tag">演练经验</span><span class="tag" :class="entry.state === 'approved' ? 'teal' : 'amber'">{{ names[entry.state] || entry.state }}</span><span v-if="referenceValidity(entry) !== 'current'" class="tag amber">{{ effectiveNames[referenceValidity(entry)] }}</span><button class="secondary small" :disabled="working" @click="openDocument(entry)">查看原文</button><button v-if="safety && entry.kind !== 'experience'" class="ghost small" :disabled="working" @click="updateDocument(entry)">更新此资料</button>
      </article><p v-if="!entries.length" class="muted">当前工作面没有符合条件的资料。</p>
    </section>
    <section v-if="loading || detail" class="panel original-panel" role="region" aria-label="资料原文">
      <div class="section-heading"><div><h2>{{ detail?.title || (detail?.kind === 'experience' ? '处置经验' : '资料原文') }}</h2><p v-if="detail">{{ detail.source }} · 版本 {{ detail.version }}</p></div><button class="ghost small" @click="close">关闭原文</button></div>
      <p v-if="loading" class="muted">正在读取原文。</p>
      <template v-if="detail"><span class="tag" :class="detail.state === 'approved' ? 'teal' : 'amber'">{{ names[detail.state] }}</span><span v-if="referenceValidity(detail) !== 'current'" class="tag amber">{{ effectiveNames[referenceValidity(detail)] }}</span><p v-if="detail.applicability?.valid_from || detail.applicability?.valid_to" class="muted">适用时间：{{ detail.applicability.valid_from ? mineTime(detail.applicability.valid_from) : '未设起点' }} 至 {{ detail.applicability.valid_to ? mineTime(detail.applicability.valid_to) + '（不含）' : '未设截止' }}</p><div v-if="fileMetadata" class="file-summary"><span>{{ fileMetadata.filename }} · {{ fileMetadata.size_bytes < 1024 ? fileMetadata.size_bytes + '字节' : (fileMetadata.size_bytes / 1024).toFixed(1) + ' KiB' }}</span><button class="secondary small" :disabled="working" @click="download">下载原件</button><small>文件校验值 {{ fileMetadata.sha256 }}</small></div>
        <article v-for="fragment in detail.fragments || []" :key="fragment.reference_id" class="original-fragment"><strong>{{ referenceLocation(fragment.locator) }}</strong><p>{{ fragment.text }}</p></article>
        <div v-if="detail.reviewer_id || detail.reviewed_by || detail.reviewed_at" class="review-record"><h3>审核记录</h3><p>{{ names[detail.state] }} · {{ detail.reviewer_id || detail.reviewed_by }} · {{ mineTime(detail.reviewed_at) }}</p><p v-if="detail.review_comment">{{ detail.review_comment }}</p></div>
        <form v-if="safety && detail.state === 'pending_review'" class="review-form" @submit.prevent="review('approved')"><label>审核说明<textarea v-model="reviewComment" rows="2"></textarea></label><div class="button-group"><button class="primary small" :disabled="working || referenceValidity(detail) !== 'current'">{{ detail.kind === 'experience' ? '批准经验' : '批准资料' }}</button><button type="button" class="ghost small" :disabled="working" @click="review('rejected')">{{ detail.kind === 'experience' ? '拒绝经验' : '拒绝资料' }}</button></div></form>
      </template>
    </section>
    <section class="panel"><div class="section-heading"><div><h2>已批准资料检索</h2><p>检索现行已批准版本，保留来源和页段定位。</p></div></div><form class="search-form" @submit.prevent="find"><input v-model="search" aria-label="资料检索关键词" placeholder="输入关键词"><button class="primary" :disabled="!session || working">检索</button></form><p v-if="results[0]?.knowledge_as_of" class="muted">资料核验时点：{{ mineTime(results[0].knowledge_as_of) }}</p><p v-if="!session" class="muted">选择会话后检索适用资料。</p><article v-for="item in results" :key="item.reference_id" class="reference-card"><span class="tag teal">{{ item.kind === 'experience' ? '已批准经验' : '已批准条文' }}</span><p>{{ item.text }}</p><small>{{ item.source }} · 版本 {{ item.version }} · {{ referenceLocation(item.locator) }}</small><button class="ghost small" :disabled="!item.entry_id && !item.document_id" @click="openDocument(item)">查看引用原文</button></article><p v-if="searched && !results.length" class="muted">没有匹配的已批准资料。</p></section>
    <section v-if="safety" class="panel"><div class="section-heading"><div><h2>{{ updatingEntry ? '更新资料版本' : '资料入库' }}</h2><p>{{ updatingEntry ? '当前版本 ' + updatingEntry.version + '，请登记新的版本编号。' : '登记来源和版本后送交审核。' }}</p></div></div><button v-if="updatingEntry" class="ghost small" :disabled="working" @click="resetForm">取消更新</button><div class="button-group import-modes"><button class="secondary small" :class="{ active: mode === 'file' }" @click="mode = 'file'">上传文档</button><button class="secondary small" :class="{ active: mode === 'text' }" @click="mode = 'text'">录入条文</button></div><form @submit.prevent="importDocument"><label v-if="mode === 'file'">资料文件<input :key="fileSelection" type="file" aria-label="资料文件" accept=".pdf,.docx,.txt,.md" @change="selectFile"><small>PDF、DOCX、TXT、MD，单个文件上限20 MiB。</small></label><div class="form-row"><label>资料名称<input v-model="form.title" required></label><label>资料来源<input v-model="form.source" required></label></div><div class="form-row"><label>资料版本<input v-model="form.version" required></label><label v-if="mode === 'text'">条文页码<input v-model="form.page" type="number" min="1" required></label></div><div class="form-row"><label>生效时间<input v-model="form.validFrom" type="datetime-local"></label><label>失效时间<input v-model="form.validTo" type="datetime-local"></label></div><p class="muted">适用期可留空；失效时间起不再适用。</p><label v-if="mode === 'text'">条文原文<textarea v-model="form.text" rows="5" required></textarea></label><button class="primary" :disabled="!faceId || working || (mode === 'file' && !selectedFile)">{{ mode === 'file' ? '上传并送审' : '登记并送审' }}</button></form></section>
  </div>
</template>

<style scoped>
.knowledge-workspace{display:grid;gap:18px;min-width:0}.state-filter{display:flex;align-items:center;gap:10px;max-width:270px;margin-bottom:10px;white-space:nowrap}.state-filter select{margin:0;min-width:0;flex:1;width:auto}.document-row{gap:10px}.document-name{flex:1;min-width:0}.document-name strong,.document-name small{overflow-wrap:anywhere}.original-panel{min-width:0}.original-fragment{padding:13px 14px;margin-top:12px;background:#102b40;border:1px solid #294a62;border-radius:4px}.original-fragment>strong{color:#80c9db;font-size:12px;font-weight:500}.original-fragment>p{white-space:pre-wrap;line-height:1.9;margin:8px 0 0;overflow-wrap:anywhere}.review-record{border-top:1px solid #294a62;padding-top:14px;margin-top:17px}.review-record p{font-size:13px;line-height:1.7}.review-form{margin-top:17px}.file-summary{display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:12px}.file-summary>small{flex-basis:100%;overflow-wrap:anywhere}.import-modes{margin-bottom:17px}.import-modes .active{border-color:#55b8d1;color:#a9e3ed}.reference-card>small{display:block;overflow-wrap:anywhere}.reference-card>button{margin-top:9px}.original-panel .section-heading>div{min-width:0}.original-panel h2,.original-panel .section-heading p{overflow-wrap:anywhere}
@media(max-width:760px){.knowledge-workspace{gap:12px}.document-row{align-items:flex-start}.document-name{flex-basis:100%}.file-summary{align-items:flex-start}.original-fragment{padding:10px}.original-panel .section-heading{align-items:flex-start}.original-panel .section-heading>button{flex-shrink:0}}
</style>
