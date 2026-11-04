// 隔离接口夹具验证界面操作；测试数据不进入产品，也不调用真实模型。
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { once } from 'node:events'
import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { resolve, extname } from 'node:path'
import test from 'node:test'
import { chromium } from 'playwright'
import { mineTime } from '../src/utils.mjs'

const root = resolve(process.env.FRONTEND_DIST_DIR || resolve(import.meta.dirname, '..', 'dist'))
const artifactDir = resolve(process.env.UI_TEST_ARTIFACT_DIR || resolve(import.meta.dirname, 'artifacts'))
const browserPath = process.env.UI_TEST_BROWSER || [
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
].find(path => existsSync(path))
const source = { source_id: 'qa-source', name: '界面验收数据（测试夹具）', face_id: 'TEST-UI' }
const otherSource = { source_id: 'qa-other-source', name: '第二来源（测试夹具）', face_id: source.face_id }
const point = { point_id: 'qa-T0', source_id: source.source_id, face_id: source.face_id, name: 'T0（界面验收）', quantity: 'methane', unit: '%CH4', unit_status: 'inferred' }
const wind = { point_id: 'qa-wind', source_id: source.source_id, face_id: source.face_id, name: '风速（界面验收）', quantity: 'wind_speed', unit: 'source_native', unit_status: 'inferred' }
const otherPoint = { ...point, point_id: 'qa-other-T0', source_id: otherSource.source_id, name: '第二来源T0' }
const modelCatalog = [
  { model_id: 'QA离线基线', version: 'test', model_type: 'baseline', default_eligible: true, point_ids: [point.point_id], input_window: 2, horizon_seconds: 3600, aggregation_seconds: 300 },
  { model_id: 'QA专业预测', version: 'trained-test-v1', model_type: 'professional', deployment_stage: 'experimental', default_eligible: false, point_ids: [point.point_id], input_window: 90, horizon_seconds: 3600, aggregation_seconds: 300 },
]
const reference = { reference_id: 'qa-approved-reference', entry_id: 'qa-document', version: '1', kind: 'document', source: 'UI验收资料', locator: { page: 1 }, text: '测试条文：核对监测记录。', operation_context: 'field_record', knowledge_as_of: '2026-11-04T12:00:00Z' }
let role, session, actions, advice, calls, pendingChat, taskState, attributionState, forecastRuns, savedSessions, nextEvidenceGate, documents, nextDocumentGate
let briefStatus, briefAttempt, briefAsOf, briefRevision, selectedEventRevision, generateOnTick, eventPresent, nextBriefGate
let evaluationReports, nextMetricGate
function reset(value) {
  role = value; session = null; actions = []; advice = []; calls = []; documents = []; nextDocumentGate = null; pendingChat = false; taskState = 'running'; attributionState = 'completed'; forecastRuns = []; savedSessions = new Map(); nextEvidenceGate = null
  briefStatus = 'not_started'; briefAttempt = 0; briefAsOf = null; briefRevision = 1; selectedEventRevision = 1; generateOnTick = false; eventPresent = true; nextBriefGate = null
  evaluationReports = []; nextMetricGate = null
}
function evaluationReport() {
  return { evaluation_id: 'qa-evaluation', manifest_id: 'qa-manifest', face_id: source.face_id, generated_at: '2026-11-04T12:00:00Z',
    records_sha256: 'b'.repeat(64), manifest_sha256: 'c'.repeat(64), report_sha256: 'd'.repeat(64),
    metric_values: { precision: 0, recall: null, mae: 0.1 }, numerators: { precision: 0, recall: 0, mae: 5.1 }, denominators: { precision: 2, recall: 0, mae: 51 },
    metric_statuses: { precision: { status: 'calculated', reason: null }, recall: { status: 'unavailable', reason: 'no_actual_events' }, mae: { status: 'calculated', reason: null } },
    scope_metadata: { source_id: source.source_id, session_id: session.session_id, point_ids: [point.point_id], start_at: session.start_as_of, end_at: session.as_of, as_of: session.as_of },
    collection_summary: { actual_events: 0, alert_events: 2, samples: 51, tasks: 1, valid_monitoring_hours: 1 },
    sample_refs: { actual_events: [], alert_events: ['qa-predicted-event'], samples: ['qa-frozen-sample'], tasks: ['qa-tool-task'] }, calibration_views: {}, missing_fields: [] }
}
function forecast(modelId = modelCatalog[0].model_id, version = modelCatalog[0].version) {
  const cutoff = Math.floor(Date.parse(session.as_of) / 300000) * 300000
  const run = { run_id: `qa-forecast-${forecastRuns.length + 1}`, session_id: session.session_id,
    source_id: session.source_id, face_id: source.face_id, point_id: point.point_id, as_of: session.as_of,
    cutoff: new Date(cutoff).toISOString(), created_at: session.as_of, model_id: modelId, model_version: version,
    status: 'completed', model_type: modelId === 'QA专业预测' ? 'professional' : 'baseline', missing_fields: [],
    input_origin: 'server_observations', input_refs: [{ id: 'qa-input' }], horizon_seconds: 3600,
    outputs: [{ prediction_stage: modelId === 'QA专业预测' ? 'residual_corrected' : 'baseline_persistence',
      quantity_semantics: 'physical_forecast', unit: '%CH4', interval_values: Array.from({ length: 12 }, (_, i) => ({
        start_at: new Date(cutoff + i * 300000).toISOString(), end_at: new Date(cutoff + (i + 1) * 300000).toISOString(), value: 0.55 + i * 0.01 })) }] }
  forecastRuns.unshift(run)
  return run
}
function event() {
  return { event_id: 'qa-event', session_id: session.session_id, source_id: session.source_id, point_id: point.point_id, event_revision: selectedEventRevision,
    start_at: new Date(Date.parse(session.as_of) - 60000).toISOString(), event_type: 'observed_exceedance',
    alert_level: 'red', quality_flags: ['suspected_calibration'] }
}
function startBrief() { briefStatus = 'queued'; briefAttempt++; briefAsOf = session.as_of; briefRevision = selectedEventRevision }
function briefPacket() {
  const asOf = briefAsOf || session.as_of
  const observation = { id: 'qa-observation', point_id: point.point_id, source_file_id: 'qa-file', row_number: 11,
    event_time: asOf, available_at: asOf, raw_value: '0.91', value: 0.91, quality_flags: [] }
  const snapshot = { ...event(), start_at: new Date(Date.parse(asOf) - 60000).toISOString(), event_revision: briefRevision, revision_as_of: asOf }
  const citations = [
    { citation_id: 'E1', kind: 'event', label: '预警事件', reference: { kind: 'event', event_id: 'qa-event', event_revision: briefRevision, revision_as_of: asOf } },
    { citation_id: 'O1', kind: 'observation', label: '实测监测记录', reference: { kind: 'observation', observation_id: observation.id, ...observation } },
    { citation_id: 'D1', kind: 'reference', label: '已审核规程 第1页', reference: { kind: 'reference', reference_id: reference.reference_id, entry_id: 'qa-document', version: '1', locator: { page: 1 }, source: reference.source } },
  ]
  const steps = [{ call_id: 'brief-evidence', name: 'get_event_evidence', arguments: { event_id: 'qa-event', event_revision: briefRevision },
    status: briefStatus === 'running' ? 'running' : 'success', elapsed_ms: 12.5 }]
  const task = briefStatus === 'not_started' ? null : {
    task_id: `qa-brief-task-${briefAttempt}`, state: briefStatus, kind: 'event_brief', method: 'deepseek', model: 'deepseek-flash',
    session_id: session.session_id, source_id: session.source_id, event_id: 'qa-event', event_revision: briefRevision, as_of: asOf,
    final_text: ['completed', 'partial'].includes(briefStatus) ? 'T0触发实测预警，监测记录与事件版本一致。[E1][O1] 参阅已审核资料。[D1]' : briefStatus === 'failed' ? '模型服务暂不可用，已保存读取结果。' : '',
    tool_steps: briefStatus === 'queued' ? [] : steps, tool_calls: briefStatus === 'queued' ? [] : steps, citations: briefStatus === 'queued' ? [] : citations, evidence_refs: briefStatus === 'queued' ? [] : citations.map(item => item.reference),
    result_cards: briefStatus === 'queued' ? [] : [
      { tool: 'get_event_evidence', status: 'success', data: { event_snapshot: snapshot, observation_fragments: [observation], evidence_refs: citations.slice(0, 2).map(item => item.reference), rules: { rule_version: 'qa-draft', observed_red_threshold: 0.8 } } },
      { tool: 'search_documents', status: 'success', data: [{ ...reference, text: undefined, entry_id: 'qa-document', version: '1', title: '界面验收规程' }] },
    ], missing_fields: [], usage: {}, error_code: briefStatus === 'failed' ? 'model_service_unavailable' : undefined,
  }
  return { event_id: 'qa-event', event_revision: briefRevision, requested_event_revision: selectedEventRevision,
    session_id: session.session_id, source_id: session.source_id, as_of: asOf, status: briefStatus,
    stale: briefRevision < selectedEventRevision, task_id: task?.task_id || null, task, attempt: briefAttempt,
    can_cancel: ['queued', 'running'].includes(briefStatus), can_retry: ['failed', 'partial', 'cancelled'].includes(briefStatus) }
}
function candidateAttribution() {
  if (attributionState === 'not_ready') return { status: 'not_ready', causal_status: 'evidence_only',
    missing_fields: ['six_channel_point_bindings', 'complete_61_minute_six_channel_history', 'automatic_attribution_enabled'] }
  if (attributionState === 'failed') return { status: 'failed', causal_status: 'evidence_only', error_code: 'model_inference_failed' }
  const targetTime = event().start_at
  return { status: 'completed', causal_status: 'candidate_attribution', model_id: 'kmeans-svc-31002',
    version: 'trained-20261103-v1', model_version: 'trained-20261103-v1', feature_version: 'legacy_row_stats_v1',
    preprocessing: 'absolute', as_of: session.as_of, input_as_of: new Date(Date.parse(targetTime) - 300000).toISOString(),
    horizon_seconds: 300, target_time: targetTime, target_observation_id: 'qa-target-observation',
    target_policy: 'first_model_threshold_observation_in_event', probability_semantics: 'cluster_membership',
    ...(attributionState === 'completed' ? { cluster_probabilities: [{ cluster_id: 0, probability: 0.62 }, { cluster_id: 2, probability: 0.38 }] } : {}),
    selected_cluster_ids: [0, 2], candidate_causes: ['割煤速度快', '地质构造影响'],
    training_range: { start_at: '2021-11-01T00:00:00+08:00', end_at: '2021-12-01T00:00:00+08:00' },
    run_id: 'qa-attribution-run', session_id: session.session_id, source_id: source.source_id, face_id: source.face_id,
    event_id: 'qa-event', event_revision: 1, input_origin: 'server_observations', input_refs: ['qa-input-1'], created_at: session.as_of }
}
function casePacket() {
  const last = actions.at(-1)
  const phase = { confirm_event: 'confirmed', review_advice: 'advice_reviewed', record_execution: 'executed', submit_feedback: 'feedback_submitted', review_case: 'reviewed' }[last?.action_type] || 'new'
  const review = actions.findLast(item => item.action_type === 'review_advice')
  return { case: { case_id: last ? 'qa-case' : null, phase, operation_context: 'rehearsal' }, actions, advice_versions: advice,
    advice_review_status: review ? { 'qa-advice:1': { decision: review.review_decision } } : {}, pending_items: [],
    current_event_revision: selectedEventRevision,
    advice_validity: Object.fromEntries(advice.map(item => [`${item.advice_id}:${item.version}`, { stale: item.event_revision !== selectedEventRevision }])),
    reviewed_snapshot: phase === 'reviewed' ? { case_id: 'qa-case', event_id: 'qa-event', event_revision: selectedEventRevision, session_id: session.session_id,
      source_id: source.source_id, face_id: source.face_id, snapshot_ref: 'qa-snapshot', reviewed_at: last.occurred_at,
      operation_context: 'rehearsal', actions: [...actions], source_refs: [reference.reference_id], advice_refs: [] } : null }
}
async function api(path, request) {
  const chunks = []
  for await (const chunk of request) chunks.push(chunk)
  const text = Buffer.concat(chunks).toString()
  const body = request.headers['content-type']?.startsWith('multipart/form-data') ? { multipart: text } : text ? JSON.parse(text) : null
  calls.push({ path: path.pathname, query: Object.fromEntries(path.searchParams), method: request.method, body, authenticated: request.headers.authorization === 'Bearer qa-isolated-token' })
  if (path.pathname === '/api/auth/login') return { access_token: 'qa-isolated-token', token_type: 'bearer', user: { user_id: 'qa-user', username: '界面验收账户', roles: [role], face_ids: [source.face_id] } }
  if (path.pathname === '/api/auth/logout') return {}
  if (path.pathname === '/api/sources') return [source, otherSource]
  if (path.pathname === '/api/points') return path.searchParams.get('source_id') === otherSource.source_id ? [otherPoint] : [point, wind]
  if (/^\/api\/sources\/[^/]+\/coverage$/.test(path.pathname)) {
    const selectedSource = path.pathname.includes(otherSource.source_id) ? otherSource : source
    const selectedPoints = selectedSource === source ? [point, wind] : [otherPoint]
    return { source_id: selectedSource.source_id, face_id: source.face_id, semantics: 'stored_history_catalog', validity_policy: 'non_null_value',
      observation_count: 2400, valid_observation_count: 2398, start_at: '2022-02-05T12:00:00+00:00', end_at: '2022-02-05T23:59:00+00:00',
      points: selectedPoints.map(p => ({ ...p, observation_count: 1200, valid_observation_count: 1199,
        start_at: '2022-02-05T12:00:00+00:00', end_at: '2022-02-05T23:59:00+00:00' })) }
  }
  if (path.pathname === '/api/references') return documents
  if (path.pathname === '/api/references/files' && request.method === 'POST') return { file_id: 'qa-upload', filename: '合成条文.txt', extension: '.txt', sha256: 'a'.repeat(64), size_bytes: 33, face_ids: [source.face_id] }
  if (path.pathname === '/api/references/files/qa-upload') return { file_id: 'qa-upload', filename: '合成条文.txt', extension: '.txt', sha256: 'a'.repeat(64), size_bytes: 33, face_ids: [source.face_id] }
  if (path.pathname === '/api/references/imports') {
    const doc = { ...body.spec, entry_id: 'qa-upload-doc', document_id: 'qa-upload-doc', kind: 'document', state: 'pending_review', fragment_count: 1,
      fragments: [{ ...reference, reference_id: 'qa-upload-reference', entry_id: 'qa-upload-doc', version: body.spec.version, locator: { paragraph: 1 }, text: '合成资料：核对监测记录并留存依据。' }] }
    documents.push(doc); return doc
  }
  if (path.pathname === '/api/references/qa-upload-doc') {
    const packet = structuredClone(documents.find(doc => doc.entry_id === 'qa-upload-doc' && String(doc.version) === path.searchParams.get('version')))
    const gate = nextDocumentGate; nextDocumentGate = null
    if (gate) await gate
    return packet
  }
  if (path.pathname === '/api/references/qa-upload-doc/review') {
    const doc = documents.find(doc => doc.entry_id === 'qa-upload-doc' && String(doc.version) === String(body.version))
    if (body.decision === 'approved') documents.filter(item => item.entry_id === doc.entry_id && item !== doc && item.state === 'approved').forEach(item => { item.state = 'superseded' })
    Object.assign(doc, { state: body.decision, reviewer_id: 'qa-user', reviewed_at: new Date().toISOString(), review_comment: body.comment }); return doc
  }
  if (path.pathname === '/api/experiences') {
    const item = { ...body.spec, entry_id: 'qa-experience', experience_id: 'qa-experience', kind: 'experience', version: 1, state: 'pending_review', source: '处置单 qa-case', operation_context: 'rehearsal',
      fragments: [{ reference_id: 'qa-experience-reference', text: body.spec.content, locator: { case_id: 'qa-case' } }] }
    documents.push(item); return item
  }
  if (path.pathname === '/api/references/qa-experience') return documents.find(doc => doc.entry_id === 'qa-experience')
  if (path.pathname === '/api/experiences/qa-experience/review') {
    const item = documents.find(doc => doc.entry_id === 'qa-experience'); item.state = body.decision; return item
  }
  if (path.pathname === '/api/evaluations') return evaluationReports
  if (path.pathname === '/api/evaluations/from-session') {
    const report = evaluationReport(); Object.assign(report.scope_metadata, body); evaluationReports.unshift(report); return report
  }
  if (path.pathname === '/api/evaluations/qa-evaluation') return evaluationReports[0]
  if (path.pathname === '/api/evaluations/qa-evaluation/export') return { report: evaluationReports[0], manifest: { manifest_id: 'qa-manifest', records_sha256: 'b'.repeat(64) } }
  if (path.pathname === '/api/evaluations/qa-evaluation/records/observations/qa-frozen-observation') return { evaluation_id: 'qa-evaluation', records_sha256: 'b'.repeat(64), kind: 'observation', record: { id: 'qa-frozen-observation', point_id: point.point_id, raw_value: '0.5', value: 0.5, source_file_id: 'qa-frozen-file', row_number: 12 } }
  if (path.pathname === '/api/evaluations/qa-evaluation/manifest') return {
    manifest_id: 'qa-manifest', model_version: 'QA离线基线:test', rule_version: 'qa-draft', preprocess_version: 'qa-preprocess', dataset_split: 'unclassified',
    scope_metadata: evaluationReports[0].scope_metadata, records_sha256: 'b'.repeat(64), manifest_sha256: 'c'.repeat(64),
    frozen_records: { actual_events: [], alert_events: [{ event_id: 'qa-predicted-event' }], samples: Array.from({ length: 51 }, (_, index) => ({ sample_id: 'qa-frozen-sample-' + index })), tasks: [{ task_id: 'qa-tool-task' }], valid_monitoring_hours: 1,
      observations: [{ id: 'qa-frozen-observation', point_id: point.point_id, raw_value: '0.5', value: 0.5, source_file_id: 'qa-frozen-file', row_number: 12 }],
      forecast_runs: [{ run_id: 'qa-frozen-run', model_id: modelCatalog[0].model_id, model_version: 'test', as_of: session.start_as_of, outputs: [] }] }, matching_config: { tolerance_seconds: 0 } }
  if (/^\/api\/evaluations\/qa-evaluation\/metrics\//.test(path.pathname)) {
    const offset = Number(path.searchParams.get('offset') || 0), metric = path.pathname.split('/').at(-1)
    const packet = { evaluation_id: 'qa-evaluation', manifest_id: 'qa-manifest', records_sha256: 'b'.repeat(64), manifest_sha256: 'c'.repeat(64), metric,
      variant: path.searchParams.get('variant') || 'primary', value: metric === 'mae' ? 0.1 : metric === 'recall' ? null : 0,
      numerator: metric === 'mae' ? 5.1 : 0, denominator: metric === 'mae' ? 51 : metric === 'recall' ? 0 : 2, total: metric === 'mae' ? 51 : 1, offset, limit: 50,
      status: metric === 'recall' ? 'unavailable' : 'calculated', reason: metric === 'recall' ? 'no_actual_events' : null,
      rows: Array.from({ length: metric === 'mae' ? offset ? 1 : 50 : 1 }, (_, i) => ({ kind: 'sample', classification: 'FP', absolute_error: 0.1, included: true,
        record: { sample_id: `qa-frozen-${offset+i}`, run_id: 'qa-frozen-run', point_id: point.point_id, target_statistic: 'max', actual_value: 0.5, predicted_value: 0.6, evidence_refs: [{ kind: 'observation', observation_id: 'qa-frozen-observation', source_file_id: 'qa-frozen-file', row_number: 12 }] },
        snapshots: { sample: { sample_id: `qa-frozen-${offset+i}`, run_id: 'qa-frozen-run', point_id: point.point_id, target_statistic: 'max', actual_value: 0.5, predicted_value: 0.6, evidence_refs: [{ kind: 'observation', observation_id: 'qa-frozen-observation', source_file_id: 'qa-frozen-file', row_number: 12 }] } } })) }
    const gate = nextMetricGate; nextMetricGate = null; if (gate) await gate
    return packet
  }
  if (path.pathname === '/api/sessions' && request.method === 'POST') {
    session = { ...body, session_id: `qa-session-${savedSessions.size + 1}`, face_id: source.face_id, as_of: body.start_as_of, state: 'paused', speed: body.speed || 1 }
    savedSessions.set(session.session_id, session)
    return session
  }
  if (path.pathname === '/api/sessions') return [...savedSessions.values()]
  if (/^\/api\/sessions\/qa-session-\d+$/.test(path.pathname)) {
    session = savedSessions.get(path.pathname.split('/').at(-1))
    return session
  }
  if (/^\/api\/sessions\/qa-session-\d+\/actions$/.test(path.pathname)) {
    session = { ...session, ...(body.action === 'set_speed' ? { speed: body.speed } : { state: body.action === 'pause' ? 'paused' : 'running' }) }
    savedSessions.set(session.session_id, session)
    return session
  }
  if (/^\/api\/sessions\/qa-session-\d+\/tick$/.test(path.pathname)) {
    session = { ...session, as_of: body.target_as_of || new Date(Date.parse(session.as_of) + 60000 * session.speed).toISOString() }
    savedSessions.set(session.session_id, session)
    if (generateOnTick && !eventPresent) { eventPresent = true; startBrief() }
    return { session, status: 'completed', forecasts: [forecast()], events: eventPresent ? [event()] : [], briefs: briefStatus === 'not_started' ? [] : [briefPacket()] }
  }
  if (path.pathname === '/api/observations') return { point_id: path.searchParams.get('point_id'), source_id: session.source_id, as_of: session.as_of,
    unit: path.searchParams.get('point_id') === wind.point_id ? 'source_native' : '%CH4', total: 3,
    quality: { valid_rows: 1, invalid_rows: 1 }, rows: [
      { event_time: new Date(Date.parse(session.as_of) - 120000).toISOString(), value: 0.91 },
      { event_time: new Date(Date.parse(session.as_of) - 60000).toISOString(), value: null },
      { event_time: new Date(Date.parse(session.as_of) + 60000).toISOString(), value: 9.99 },
    ] }
  if (path.pathname === '/api/events') return !eventPresent || path.searchParams.get('point_id') === wind.point_id ? [] : [event()]
  if (path.pathname === '/api/events/qa-event/brief/cancel') { briefStatus = 'cancelled'; return briefPacket() }
  if (path.pathname === '/api/events/qa-event/brief') {
    if (request.method === 'POST') {
      if (briefStatus === 'not_started' || body.retry || briefRevision < selectedEventRevision) startBrief()
      return briefPacket()
    }
    const packet = briefPacket(), gate = nextBriefGate; nextBriefGate = null
    if (gate) await gate
    return packet
  }
  if (path.pathname === '/api/rules') return { rule_version: 'qa-draft', status: 'team-draft', source: 'UI验收规则（测试夹具）', observed_red_threshold: 0.8, observed_orange_threshold: 0.65, orange_duration_seconds: 300, forecast_yellow_threshold: 0.65 }
  if (path.pathname === '/api/models') return path.searchParams.get('point_id') === wind.point_id ? [] : modelCatalog.map(m => ({ ...m, source_id: session.source_id, face_id: source.face_id }))
  if (path.pathname === '/api/forecasts' && request.method === 'POST') return forecast(body.model_id, body.version)
  if (path.pathname === '/api/forecasts') return forecastRuns.filter(run => run.session_id === path.searchParams.get('session_id')
    && (!path.searchParams.has('point_id') || run.point_id === path.searchParams.get('point_id'))
    && (!path.searchParams.has('model_id') || run.model_id === path.searchParams.get('model_id'))
    && (!path.searchParams.has('version') || run.model_version === path.searchParams.get('version'))).slice(0, Number(path.searchParams.get('limit') || 20))
  if (/^\/api\/forecasts\/qa-forecast-\d+$/.test(path.pathname)) return forecastRuns.find(run => path.pathname.endsWith(run.run_id))
  if (path.pathname === '/api/evidence') {
    const packet = { event_snapshot: event(), snapshot_as_of: session.as_of,
      observation_fragments: [], condition_hints: [{ kind: 'suspected_calibration', text: 'UI验收：疑似标校提示。' }], missing_fields: [],
      candidate_attribution: candidateAttribution() }
    const gate = nextEvidenceGate; nextEvidenceGate = null
    if (gate) await gate
    return packet
  }
  if (path.pathname === '/api/cases/qa-event') return casePacket()
  if (path.pathname === '/api/cases/qa-event/actions') {
    actions.push({ ...body, action_id: `qa-action-${actions.length}`, actor_id: 'qa-user', occurred_at: new Date().toISOString(), data_as_of: session.as_of })
    return { case: casePacket().case, case_action: actions.at(-1) }
  }
  if (path.pathname === '/api/references/search') return [reference]
  if (path.pathname === '/api/references/qa-document') return { entry_id: 'qa-document', document_id: 'qa-document', version: '1', title: '界面验收规程', state: 'approved', source: reference.source,
    fragments: [{ ...reference, entry_id: 'qa-document', version: '1' }] }
  if (path.pathname === '/api/advice') {
    advice.push({ advice_id: 'qa-advice', version: 1, event_revision: selectedEventRevision, knowledge_as_of: reference.knowledge_as_of, data_as_of: session.as_of, reference_ids: body.reference_ids, items: [reference] })
    return advice.at(-1)
  }
  if (path.pathname === '/api/chat') return { task_id: 'qa-task', state: pendingChat ? 'running' : 'completed', final_text: pendingChat ? '' : 'UI验收：工具任务已完成，数据截止当前会话时点。', brief_method: 'structured_template' }
  if (path.pathname === '/api/tasks/qa-task/cancel') { taskState = 'cancelled'; return { task_id: 'qa-task', state: taskState } }
  if (path.pathname === '/api/tasks/qa-task') return { task_id: 'qa-task', state: taskState, final_text: taskState === 'cancelled' ? 'UI验收：任务已取消。' : '' }
  throw new Error(`测试夹具未定义接口 ${path.pathname}`)
}

test('headless UI acceptance with isolated API fixtures', { timeout: 90000 }, async t => {
  assert.ok(browserPath, '设置 UI_TEST_BROWSER 指向已安装的 Chromium/Edge/Chrome 可执行文件')
  const server = createServer(async (request, response) => {
    const path = new URL(request.url, 'http://127.0.0.1')
    try {
      if (path.pathname.startsWith('/api/')) {
        if (path.pathname === '/api/references/files/qa-upload/content') {
          calls.push({ path: path.pathname, method: request.method, authorization: request.headers.authorization })
          response.setHeader('Content-Type', 'text/plain; charset=utf-8')
          response.end('合成资料：核对监测记录并留存依据。'); return
        }
        response.setHeader('Content-Type', 'application/json; charset=utf-8')
        response.end(JSON.stringify(await api(path, request)))
      } else {
        const relative = path.pathname === '/' ? 'index.html' : path.pathname.slice(1)
        const file = resolve(root, relative)
        if (!file.startsWith(root)) { response.writeHead(403); response.end(); return }
        response.setHeader('Content-Type', { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' }[extname(file)] || 'application/octet-stream')
        response.end(await readFile(file))
      }
    } catch (error) { response.writeHead(500, { 'Content-Type': 'application/json' }); response.end(JSON.stringify({ message: error.message })) }
  })
  server.listen(0, '127.0.0.1'); await once(server, 'listening')
  const url = `http://127.0.0.1:${server.address().port}`
  const browser = await chromium.launch({ executablePath: browserPath, headless: true })
  const externalRequests = []
  async function monitoredPage(context) {
    const page = await context.newPage()
    page.on('request', request => { if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== url) externalRequests.push(request.url()) })
    return page
  }
  try {
    for (const userRole of ['ventilation', 'gas_governance', 'safety_management']) {
      await t.test(`${userRole}: login, scoped data, chart and operation permissions`, async () => {
        reset(userRole)
        const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } })
        try {
        const page = await monitoredPage(context)
        page.setDefaultTimeout(6000)
        const failures = []; page.on('pageerror', error => failures.push(error.message))
        await page.clock.install()
        await page.goto(url)
        assert.equal(calls.length, 0, '登录前不请求业务资料')
        await page.locator('[name="username"]').fill('qa-only-user')
        await page.locator('[name="password"]').fill('qa-only-password-for-test')
        await page.getByRole('button', { name: /^登录工作台/ }).click()
        await page.getByRole('heading', { name: 'TEST-UI 工作面', exact: true }).waitFor()
        await page.getByRole('button', { name: '创建会话', exact: true }).click()
        assert.equal(await page.getByRole('region', { name: '综合态势', exact: true }).count(), 1, '正式工作台采用综合态势布局')
        assert.equal(await page.getByRole('combobox', { name: '预测模型', exact: true }).count(), 1, '预测模型可选择')
        assert.equal(await page.getByRole('region', { name: '监测曲线', exact: true }).count(), 1, '曲线提供独立的可访问区域')
        const overview = page.getByRole('region', { name: '综合态势', exact: true })
        const curve = page.getByRole('region', { name: '监测曲线', exact: true })
        await overview.getByText('0.910', { exact: true }).first().waitFor()
        assert.ok(calls.filter(call => call.path === '/api/observations').every(call => call.query.end_at === session.as_of))
        assert.equal(await overview.getByText('9.990', { exact: true }).count(), 0, '未来读数不出现在通道卡片')
        assert.equal(await page.getByRole('button', { name: '批准建议', exact: true }).count(), 0)
        assert.equal(calls.some(call => call.path === '/api/evaluations'), userRole === 'safety_management')
        await page.getByRole('combobox', { name: '预测模型', exact: true }).selectOption('QA专业预测:trained-test-v1')
        await page.getByRole('button', { name: '运行预测', exact: true }).click()
        await curve.getByText('trained-test-v1', { exact: true }).first().waitFor()
        const forecastTable = curve.locator('details').filter({ has: page.locator('summary', { hasText: '预测区间数据' }) })
        await forecastTable.locator('summary').click()
        await forecastTable.getByRole('cell', { name: '0.550', exact: true }).waitFor()
        await forecastTable.getByRole('cell', { name: '0.660', exact: true }).waitFor()
        assert.equal(await forecastTable.locator('tbody tr').count(), 12)
        const submittedForecast = calls.findLast(call => call.path === '/api/forecasts' && call.method === 'POST')
        assert.deepEqual(submittedForecast.body, { session_id: session.session_id, point_id: point.point_id, model_id: 'QA专业预测', version: 'trained-test-v1' })
        await forecastTable.locator('summary').click()
        await page.getByRole('button', { name: /实测超限/ }).click()
        assert.equal(await page.getByRole('tab', { name: '研判简报', exact: true }).count(), 1, '事件工作区提供主动研判简报')
        await curve.getByText('trained-test-v1', { exact: true }).first().waitFor()
        assert.equal(calls.findLast(call => call.path === '/api/evidence').query.event_revision, '1', '依据查询绑定事件修订')
        const attributionCard = page.getByRole('region', { name: '溯源结果', exact: true })
        await attributionCard.getByText('已完成', { exact: true }).waitFor()
        const attributionText = await attributionCard.innerText()
        for (const text of ['候选原因', '割煤速度快', '地质构造影响', '模型版本', 'kmeans-svc-31002', 'trained-20261103-v1',
          '目标时间', mineTime(candidateAttribution().target_time), '输入截止', mineTime(candidateAttribution().input_as_of), '模型提前量']) {
          assert.ok(attributionText.includes(text), `溯源卡展示 ${text}`)
        }
        await attributionCard.locator('summary', { hasText: '类别概率' }).click()
        await attributionCard.getByRole('cell', { name: '62.0%', exact: true }).waitFor()
        await attributionCard.getByRole('cell', { name: '38.0%', exact: true }).waitFor()
        if (userRole === 'ventilation') {
          attributionState = 'not_ready'
          await page.getByRole('button', { name: /实测超限/ }).click()
          await attributionCard.getByText('待补充', { exact: true }).waitFor()
          await attributionCard.getByText('补齐六路测点映射', { exact: true }).waitFor()
          await attributionCard.getByText('补齐61个连续分钟时刻的六路数据', { exact: true }).waitFor()
          await attributionCard.getByText('当前模型未启用事件溯源。', { exact: true }).waitFor()
          assert.equal(await attributionCard.getByText(/割煤速度快|six_channel_point_bindings/).count(), 0)
          attributionState = 'failed'
          await page.getByRole('button', { name: /实测超限/ }).click()
          await attributionCard.getByText('分析未完成', { exact: true }).waitFor()
          assert.equal(await attributionCard.getByText(/model_inference_failed/).count(), 0)
          attributionState = 'completed_without_probabilities'
          await page.getByRole('button', { name: /实测超限/ }).click()
          await attributionCard.getByText('已完成', { exact: true }).waitFor()
          assert.equal(await attributionCard.locator('summary', { hasText: '类别概率' }).count(), 0)
        }
        await page.getByRole('button', { name: '打开人工处置', exact: true }).click()
        await page.getByPlaceholder('填写人工核实、执行、反馈或复核记录').waitFor()
        assert.equal(await page.getByRole('button', { name: '批准建议', exact: true }).count(), userRole === 'safety_management' ? 1 : 0)
        if (userRole === 'safety_management') {
          await page.clock.runFor(15000)
          assert.equal(calls.filter(call => call.path.endsWith('/tick')).length, 0, '暂停时自动回放不推进')
          await page.getByRole('button', { name: '▷ 继续', exact: true }).click()
          await page.locator('.session-strip .muted').filter({ hasText: '进行中' }).waitFor()
          await Promise.all([page.waitForResponse(response => response.url().endsWith('/tick')), page.clock.runFor(5001)])
          await page.getByRole('button', { name: 'Ⅱ 暂停', exact: true }).click()
          await page.locator('.session-strip .muted').filter({ hasText: '已暂停' }).waitFor()
          const tickCount = calls.filter(call => call.path.endsWith('/tick')).length
          await page.clock.runFor(15000)
          assert.equal(calls.filter(call => call.path.endsWith('/tick')).length, tickCount)
          assert.deepEqual(calls.find(call => call.path.endsWith('/tick')).body, {}, '自动推进交由服务端驱动')
          const before = Date.parse(session.as_of)
          await page.getByRole('button', { name: '前进 60 秒 →', exact: true }).click()
          await page.waitForFunction(() => !document.querySelector('.loading-note'))
          assert.equal(Date.parse(session.as_of) - before, 60000)
          await page.getByPlaceholder('填写人工核实、执行、反馈或复核记录').fill('UI验收：人工确认')
          await page.getByRole('button', { name: '确认事件', exact: true }).click()
          await page.locator('.evidence-grid strong').filter({ hasText: '已确认' }).waitFor()
          await page.getByRole('textbox', { name: '建议资料关键词' }).fill('测试条文')
          await page.locator('.subsection .search-form').getByRole('button', { name: '检索', exact: true }).click()
          await page.getByRole('checkbox').check()
          await page.getByRole('button', { name: '生成带出处建议', exact: true }).click()
          await page.getByRole('button', { name: '批准建议', exact: true }).click()
          await page.waitForFunction(() => document.querySelector('.advice-items') && !document.querySelector('.loading-note'))
          assert.deepEqual(calls.find(call => call.path === '/api/advice').body.reference_ids, ['qa-approved-reference'])
          for (const [button, content] of [['记录执行', 'UI验收：执行'], ['提交反馈', 'UI验收：反馈'], ['完成复核', 'UI验收：复核']]) {
            await page.getByPlaceholder('填写人工核实、执行、反馈或复核记录').fill(content)
            await page.getByRole('button', { name: button, exact: true }).click()
            await page.waitForFunction(() => !document.querySelector('.loading-note'))
          }
          assert.equal(casePacket().case.phase, 'reviewed')
          assert.ok(actions.every(item => item.operation_context === 'rehearsal'))
          await page.getByRole('region', { name: '复核归档', exact: true }).getByText('已冻结归档', { exact: true }).waitFor()
          await page.getByRole('textbox', { name: '经验内容', exact: true }).fill('合成演练经验：保留监测依据及复核记录。')
          await page.getByRole('button', { name: '前进 60 秒 →', exact: true }).click()
          await page.waitForFunction(() => !document.querySelector('.loading-note'))
          assert.equal(await page.getByRole('textbox', { name: '经验内容', exact: true }).inputValue(), '合成演练经验：保留监测依据及复核记录。', '同一处置普通刷新保留未提交经验')
          await page.getByRole('button', { name: '提交经验审核', exact: true }).click()
          await page.getByRole('status').filter({ hasText: '经验已提交，审核后可检索。' }).waitFor()
          assert.equal(calls.findLast(call => call.path === '/api/experiences').body.case_id, 'qa-case')
          for (const width of [1440, 390]) {
            await page.setViewportSize({ width, height: 1050 })
            assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `闭环页面${width}px无整页溢出`)
            await mkdir(artifactDir, { recursive: true })
            await page.screenshot({ path: resolve(artifactDir, `case-archive-${width}.png`), fullPage: true })
          }
          await page.setViewportSize({ width: 1440, height: 1050 })
          await page.getByRole('button', { name: /资料与依据/ }).click()
          await page.getByRole('article').filter({ has: page.getByText('处置经验', { exact: true }) }).getByRole('button', { name: '查看原文', exact: true }).click()
          await page.getByRole('button', { name: '批准经验', exact: true }).click()
          await page.getByRole('region', { name: '资料原文', exact: true }).getByText('已批准', { exact: true }).first().waitFor()
          assert.equal(documents.find(item => item.entry_id === 'qa-experience').state, 'approved')
          await page.getByRole('textbox', { name: '向风险管控助手提问' }).fill('UI验收问题')
          await page.getByRole('button', { name: '发送 ↑', exact: true }).click()
          await page.getByText('UI验收：工具任务已完成，数据截止当前会话时点。', { exact: true }).waitFor()
          assert.equal(calls.find(call => call.path === '/api/chat').body.async_execution, true)
          pendingChat = true
          await page.getByRole('textbox', { name: '向风险管控助手提问' }).fill('UI验收取消任务')
          await page.getByRole('button', { name: '发送 ↑', exact: true }).click()
          await page.getByRole('button', { name: '取消任务', exact: true }).click()
          await page.clock.runFor(1001)
          await page.getByText('UI验收：任务已取消。', { exact: true }).waitFor()
          assert.ok(calls.some(call => call.path === '/api/tasks/qa-task/cancel'))
          await page.getByRole('button', { name: /模型与评估/ }).click()
          await page.getByText('离线基线', { exact: true }).waitFor()
          await page.getByRole('button', { name: /监测与预警/ }).click()
          await page.clock.runFor(100)
          const chartBounds = await page.getByRole('img', { name: '当前测点实测与预测曲线', exact: true }).evaluate(element => ({ parent: element.clientWidth, canvas: element.querySelector('canvas')?.clientWidth }))
          assert.ok(chartBounds.canvas > chartBounds.parent * 0.9, '曲线画布覆盖可用宽度')
          await mkdir(artifactDir, { recursive: true })
          await attributionCard.screenshot({ path: resolve(artifactDir, 'ui-attribution-test-fixture.png') })
          await page.evaluate(() => scrollTo(0, 0))
          await page.screenshot({ path: resolve(artifactDir, 'ui-acceptance-test-fixture.png') })
          await page.getByText('创建新会话', { exact: true }).click()
          await page.locator('.create-session select').selectOption('realtime')
          await page.getByRole('button', { name: '创建会话', exact: true }).click()
          await page.getByRole('button', { name: '刷新现场时间', exact: true }).click()
          await page.waitForFunction(() => !document.querySelector('.loading-note'))
          assert.deepEqual(calls.findLast(call => call.path.endsWith('/tick')).body, {}, '现场刷新使用服务端时间')
          await page.getByRole('button', { name: /实测超限/ }).click()
          await page.setViewportSize({ width: 390, height: 844 })
          await page.clock.runFor(100)
          assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), '窄屏无整页水平溢出')
          await attributionCard.screenshot({ path: resolve(artifactDir, 'ui-attribution-mobile-test-fixture.png') })
          await page.screenshot({ path: resolve(artifactDir, 'ui-mobile-test-fixture.png') })
        }
        assert.deepEqual(failures, [], '浏览器无运行时异常')
        } finally { await context.close() }
      })
    }
    await t.test('local knowledge upload, review and original fragments stay traceable', async () => {
      reset('safety_management')
      const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } })
      let releaseDocument = () => {}
      try {
        const page = await monitoredPage(context); page.setDefaultTimeout(6000)
        const errors = []; page.on('pageerror', error => errors.push(error.message))
        await page.goto(url)
        await page.locator('[name="username"]').fill('qa-only-user')
        await page.locator('[name="password"]').fill('qa-only-password-for-test')
        await page.getByRole('button', { name: /^登录工作台/ }).click()
        await page.getByRole('heading', { name: 'TEST-UI 工作面', exact: true }).waitFor()
        await page.getByRole('button', { name: /资料与依据/ }).click()
        await page.getByLabel('资料文件', { exact: true }).setInputFiles({ name: '合成条文.txt', mimeType: 'text/plain', buffer: Buffer.from('合成资料：核对监测记录并留存依据。') })
        await page.getByRole('textbox', { name: '资料名称', exact: true }).fill('合成资料上传验收')
        await page.getByRole('textbox', { name: '资料来源', exact: true }).fill('隔离合成资料')
        await page.getByLabel('生效时间', { exact: true }).fill('2026-01-01T00:00')
        await page.getByLabel('失效时间', { exact: true }).fill('2027-01-01T00:00')
        await page.getByRole('button', { name: '上传并送审', exact: true }).click()
        const row = page.getByRole('article').filter({ has: page.getByText('合成资料上传验收', { exact: true }) })
        await row.getByText('待审核', { exact: true }).waitFor()
        assert.ok(calls.find(call => call.path === '/api/references/files').body.multipart.includes('合成条文.txt'))
        assert.equal(calls.find(call => call.path === '/api/references/imports').body.spec.file_id, 'qa-upload')
        assert.equal(calls.find(call => call.path === '/api/references/imports').body.spec.applicability.valid_from, '2025-12-31T16:00:00.000Z')
        assert.equal(calls.find(call => call.path === '/api/references/imports').body.spec.applicability.valid_to, '2026-12-31T16:00:00.000Z')
        await row.getByRole('button', { name: '查看原文', exact: true }).click()
        const detail = page.getByRole('region', { name: '资料原文', exact: true })
        await detail.getByText('合成资料：核对监测记录并留存依据。', { exact: true }).waitFor()
        await detail.getByText('第1段', { exact: true }).waitFor()
        assert.equal(calls.findLast(call => call.path === '/api/references/qa-upload-doc').query.version, '1')
        await detail.getByRole('textbox', { name: '审核说明', exact: true }).fill('合成上传内容与原件一致。')
        await detail.getByRole('button', { name: '批准资料', exact: true }).click()
        await detail.getByText('已批准', { exact: true }).first().waitFor()
        await detail.getByText('合成上传内容与原件一致。', { exact: true }).waitFor()
        const [download] = await Promise.all([page.waitForEvent('download'), detail.getByRole('button', { name: '下载原件', exact: true }).click()])
        await mkdir(artifactDir, { recursive: true })
        const downloadPath = resolve(artifactDir, 'knowledge-original-download.txt')
        await download.saveAs(downloadPath)
        assert.equal(await readFile(downloadPath, 'utf8'), '合成资料：核对监测记录并留存依据。')
        assert.equal(calls.findLast(call => call.path.endsWith('/content')).authorization, 'Bearer qa-isolated-token')
        for (const width of [1920, 1440, 1280, 390]) {
          await page.setViewportSize({ width, height: 1050 })
          assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `知识页面${width}px无整页溢出`)
          await mkdir(artifactDir, { recursive: true })
          await page.screenshot({ path: resolve(artifactDir, `knowledge-local-${width}.png`), fullPage: true })
        }
        await page.setViewportSize({ width: 1440, height: 1050 })
        await row.getByRole('button', { name: '更新此资料', exact: true }).click()
        await page.getByRole('textbox', { name: '资料版本', exact: true }).fill('2')
        await page.getByLabel('资料文件', { exact: true }).setInputFiles({ name: '合成条文-v2.txt', mimeType: 'text/plain', buffer: Buffer.from('合成资料第二版。') })
        await page.getByRole('button', { name: '上传并送审', exact: true }).click()
        await detail.getByRole('button', { name: '批准资料', exact: true }).waitFor()
        assert.equal(calls.findLast(call => call.path === '/api/references/imports').body.spec.entry_id, 'qa-upload-doc')
        await detail.getByRole('button', { name: '批准资料', exact: true }).click()
        await page.locator('.document-row .tag').filter({ hasText: '已替代' }).first().waitFor()
        await page.getByRole('article').filter({ has: page.getByText(/隔离合成资料 · 版本 1/) }).getByRole('button', { name: '查看原文', exact: true }).click()
        await detail.getByText('已替代', { exact: true }).first().waitFor()
        assert.equal(calls.findLast(call => call.path === '/api/references/qa-upload-doc').query.version, '1', '旧版本仍可精确追溯')
        await detail.getByRole('button', { name: '关闭原文', exact: true }).click()
        nextDocumentGate = new Promise(resolveGate => { releaseDocument = resolveGate })
        await Promise.all([page.waitForRequest(request => request.url().includes('/references/qa-upload-doc?')), row.first().getByRole('button', { name: '查看原文', exact: true }).click()])
        await Promise.all([page.waitForResponse(response => response.url().includes(`/points?source_id=${otherSource.source_id}`)), page.locator('.context-selects select').first().selectOption(otherSource.source_id)])
        releaseDocument()
        assert.equal(await detail.count(), 0, '同工作面的来源切换后迟到原文不覆盖当前页面')
        assert.deepEqual(errors, [])
      } finally { releaseDocument(); await context.close() }
    })

    await t.test('event revisions prevent executing stale approved advice', async () => {
      reset('safety_management'); selectedEventRevision = 2
      advice = [{ advice_id: 'qa-advice', version: 1, event_revision: 1, items: [reference], reference_ids: [reference.reference_id] }]
      actions = [{ action_id: 'qa-confirmed', action_type: 'confirm_event', content: '合成确认', occurred_at: '2022-02-05T13:12:00Z', data_as_of: '2022-02-05T13:12:00Z' },
        { action_id: 'qa-approved', action_type: 'review_advice', review_decision: 'approved', advice_id: 'qa-advice', advice_version: 1, occurred_at: '2022-02-05T13:12:00Z', data_as_of: '2022-02-05T13:12:00Z' }]
      const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } })
      try {
        const page = await monitoredPage(context); page.setDefaultTimeout(6000)
        await page.goto(url)
        await page.locator('[name="username"]').fill('qa-only-user'); await page.locator('[name="password"]').fill('qa-only-password-for-test')
        await page.getByRole('button', { name: /^登录工作台/ }).click()
        await page.getByRole('heading', { name: 'TEST-UI 工作面', exact: true }).waitFor()
        await page.getByRole('button', { name: '创建会话', exact: true }).click()
        await page.getByRole('button', { name: /实测超限/ }).click()
        await page.getByRole('button', { name: '打开人工处置', exact: true }).click()
        await page.getByText('建议依据已更新或失效，请重新生成并审核。', { exact: true }).waitFor()
        await page.getByPlaceholder('填写人工核实、执行、反馈或复核记录').fill('合成执行说明')
        assert.equal(await page.getByRole('button', { name: '记录执行', exact: true }).isDisabled(), true)
        assert.equal(await page.getByRole('button', { name: '批准建议', exact: true }).isDisabled(), true)
        assert.equal(calls.some(call => call.path.endsWith('/actions') && call.body.action_type === 'record_execution'), false)
      } finally { await context.close() }
    })

    await t.test('forecast restore, non-methane point and stale evidence stay scoped', async () => {
      reset('safety_management')
      const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })
      let releaseEvidence = () => {}
      try {
        const page = await monitoredPage(context)
        page.setDefaultTimeout(6000)
        await page.goto(url)
        await page.locator('[name="username"]').fill('qa-only-user')
        await page.locator('[name="password"]').fill('qa-only-password-for-test')
        await page.getByRole('button', { name: /^登录工作台/ }).click()
        await page.getByRole('heading', { name: 'TEST-UI 工作面', exact: true }).waitFor()
        await page.getByRole('button', { name: '创建会话', exact: true }).click()
        const curve = page.getByRole('region', { name: '监测曲线', exact: true })
        await page.getByRole('combobox', { name: '预测模型', exact: true }).selectOption('QA专业预测:trained-test-v1')
        await page.getByRole('button', { name: '运行预测', exact: true }).click()
        await curve.getByText('trained-test-v1', { exact: true }).first().waitFor()
        const originalSession = session.session_id
        const runCount = calls.filter(call => call.path === '/api/forecasts' && call.method === 'POST').length
        nextEvidenceGate = new Promise(resolveGate => { releaseEvidence = resolveGate })
        await Promise.all([
          page.waitForRequest(request => request.url().includes('/api/evidence?')),
          page.getByRole('button', { name: /实测超限/ }).click(),
        ])
        await page.getByRole('combobox', { name: '监测点', exact: true }).selectOption(wind.point_id)
        releaseEvidence()
        await page.getByRole('heading', { name: '工况趋势', exact: true }).waitFor()
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        assert.equal(await page.getByRole('region', { name: '溯源结果', exact: true }).count(), 0, '旧事件响应不回写新测点')
        assert.equal(await page.getByRole('button', { name: '运行预测', exact: true }).isDisabled(), true)
        assert.equal(await curve.getByText('trained-test-v1', { exact: true }).count(), 0)
        assert.equal(calls.filter(call => call.path === '/api/rules' && call.query.point_id === wind.point_id).length, 0)
        assert.equal(calls.filter(call => call.path === '/api/forecasts' && call.body?.point_id === wind.point_id).length, 0)
        await page.getByRole('combobox', { name: '数据源', exact: true }).selectOption(otherSource.source_id)
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        assert.equal(await page.getByRole('button', { name: '运行预测', exact: true }).isDisabled(), true, '切来源先清空会话')
        await page.getByRole('combobox', { name: '已有会话', exact: true }).selectOption(originalSession)
        await page.getByRole('button', { name: '运行预测', exact: true }).waitFor({ state: 'visible' })
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        await page.getByRole('combobox', { name: '预测模型', exact: true }).selectOption('QA专业预测:trained-test-v1')
        await curve.getByText('trained-test-v1', { exact: true }).first().waitFor()
        assert.equal(calls.filter(call => call.path === '/api/forecasts' && call.method === 'POST').length, runCount, '恢复已有结果无需再次运行')
        assert.equal(await page.getByRole('combobox', { name: '数据源', exact: true }).inputValue(), source.source_id)
      } finally { releaseEvidence(); await context.close() }
    })
    await t.test('new event brief polls real steps, opens citations and preserves retry scope', async () => {
      reset('safety_management'); generateOnTick = true; eventPresent = false
      const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })
      let releaseBrief = () => {}
      let releaseUiEvidence = () => {}
      try {
        const page = await monitoredPage(context)
        page.setDefaultTimeout(6000)
        await page.clock.install()
        const errors = []; page.on('pageerror', error => errors.push(error.message))
        await page.goto(url)
        await page.locator('[name="username"]').fill('qa-only-user')
        await page.locator('[name="password"]').fill('qa-only-password-for-test')
        await page.getByRole('button', { name: /^登录工作台/ }).click()
        await page.getByRole('heading', { name: 'TEST-UI 工作面', exact: true }).waitFor()
        await page.getByRole('button', { name: '创建会话', exact: true }).click()
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        assert.equal(calls.filter(call => call.path.endsWith('/brief') && call.method === 'POST').length, 0)
        nextEvidenceGate = new Promise(resolveGate => { releaseUiEvidence = resolveGate })
        await page.getByRole('button', { name: '前进 60 秒 →', exact: true }).click()
        const briefPanel = page.getByRole('region', { name: '研判简报', exact: true })
        await briefPanel.getByText('排队中', { exact: true }).waitFor()
        assert.ok(await page.locator('.loading-note').count(), '慢速原始依据仍在读取时简报状态已显示')
        releaseUiEvidence()
        assert.equal(calls.filter(call => call.path.endsWith('/brief') && call.method === 'POST').length, 0, '新预警由服务端触发简报，前端不重复付费提交')
        briefStatus = 'running'
        await page.clock.runFor(1100)
        const steps = briefPanel.getByRole('region', { name: '工具调用过程', exact: true })
        await steps.getByText('1. 读取事件依据', { exact: true }).waitFor()
        await steps.getByText('执行中', { exact: true }).waitFor()
        briefStatus = 'completed'
        await page.clock.runFor(1100)
        await briefPanel.getByRole('button', { name: '正文依据 E1', exact: true }).waitFor()
        await steps.getByText('耗时 12.5 ms', { exact: true }).waitFor()
        await briefPanel.getByRole('button', { name: '打开依据 O1', exact: true }).click()
        const citation = briefPanel.getByRole('region', { name: '引用依据', exact: true })
        await citation.getByRole('cell', { name: '0.91', exact: true }).first().waitFor()
        assert.ok((await citation.innerText()).includes('qa-file'))
        await citation.getByRole('button', { name: '关闭依据', exact: true }).click()
        await briefPanel.getByRole('button', { name: '打开依据 D1', exact: true }).click()
        await citation.getByText(reference.text, { exact: true }).waitFor()
        assert.equal(calls.findLast(call => call.path === '/api/references/qa-document').query.version, '1', '资料引用打开冻结版本')
        await citation.getByRole('button', { name: '关闭依据', exact: true }).click()
        await mkdir(artifactDir, { recursive: true })
        await page.screenshot({ path: resolve(artifactDir, 'event-brief-completed.png'), fullPage: true })
        selectedEventRevision = 2
        await page.getByRole('button', { name: '刷新监测', exact: true }).click()
        await briefPanel.getByText('基于第1版 · 当前第2版', { exact: true }).waitFor()
        assert.equal(calls.filter(call => call.path.endsWith('/brief') && call.method === 'POST').length, 0, '普通修订复用已有简报')
        await briefPanel.getByRole('button', { name: '生成当前版本', exact: true }).click()
        assert.equal(calls.findLast(call => call.path.endsWith('/brief') && call.method === 'POST').body.event_revision, 2)
        briefStatus = 'failed'
        await page.clock.runFor(1100)
        await briefPanel.getByText('生成失败', { exact: true }).waitFor()
        await steps.getByText('1. 读取事件依据', { exact: true }).waitFor()
        await briefPanel.getByRole('button', { name: '重试简报', exact: true }).click()
        assert.equal(calls.findLast(call => call.path.endsWith('/brief') && call.method === 'POST').body.retry, true)
        await briefPanel.getByRole('button', { name: '取消简报', exact: true }).click()
        await briefPanel.getByText('已取消', { exact: true }).first().waitFor()
        assert.ok(calls.some(call => call.path.endsWith('/brief/cancel')))
        await briefPanel.getByRole('button', { name: '重试简报', exact: true }).click()
        nextBriefGate = new Promise(resolveGate => { releaseBrief = resolveGate })
        await Promise.all([
          page.waitForRequest(request => request.url().includes('/brief?')),
          briefPanel.getByRole('button', { name: '刷新简报', exact: true }).click(),
        ])
        await page.getByRole('combobox', { name: '监测点', exact: true }).selectOption(wind.point_id)
        releaseBrief()
        await page.getByRole('heading', { name: '工况趋势', exact: true }).waitFor()
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        const reads = calls.filter(call => call.path.endsWith('/brief') && call.method === 'GET').length
        await page.clock.runFor(5100)
        assert.equal(calls.filter(call => call.path.endsWith('/brief') && call.method === 'GET').length, reads, '切测点后停止旧简报轮询')
        assert.equal(await briefPanel.count(), 0, '迟到简报不回写工况页面')
        assert.deepEqual(errors, [])
      } finally { releaseBrief(); releaseUiEvidence(); await context.close() }
    })
    await t.test('evaluation freezes current session and drills into paged frozen samples', async () => {
      reset('safety_management')
      const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: true })
      let releaseMetric = () => {}
      try {
        const page = await monitoredPage(context); page.setDefaultTimeout(6000)
        await page.goto(url)
        await page.locator('[name="username"]').fill('qa-only-user')
        await page.locator('[name="password"]').fill('qa-only-password-for-test')
        await page.getByRole('button', { name: /^登录工作台/ }).click()
        await page.getByRole('heading', { name: 'TEST-UI 工作面', exact: true }).waitFor()
        await page.getByRole('button', { name: '创建会话', exact: true }).click()
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        await page.getByRole('button', { name: '前进 60 秒 →', exact: true }).click()
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        await page.getByRole('button', { name: '模型与评估', exact: false }).click()
        await page.getByRole('button', { name: '冻结当前会话评估', exact: true }).click()
        const report = page.getByRole('region', { name: '冻结评估详情', exact: true })
        await report.getByText('没有实测阈值事件', { exact: true }).waitFor()
        assert.equal(calls.findLast(call => call.path === '/api/evaluations/from-session').body.dataset_split, 'unclassified')
        assert.equal(calls.findLast(call => call.path === '/api/evaluations/from-session').body.model_version, 'test')
        const liveReads = calls.filter(call => ['/api/evidence','/api/observations'].includes(call.path)).length
        await report.getByRole('button', { name: '查看平均绝对误差明细', exact: true }).click()
        const details = page.getByRole('region', { name: '指标明细', exact: true })
        await details.getByText('共51条', { exact: true }).waitFor()
        await details.getByRole('button', { name: '查看冻结记录 1', exact: true }).click()
        const frozen = page.getByRole('region', { name: '冻结记录', exact: true })
        await frozen.getByText('qa-frozen-0', { exact: true }).waitFor()
        assert.equal(await frozen.getByText('实测均值', { exact: true }).count(), 0, '最大值口径不能标为实测均值')
        await frozen.getByText('实测统计值', { exact: true }).waitFor()
        await frozen.getByText('最大值', { exact: true }).waitFor()
        assert.equal(await page.getByText(/有效实测均值配对/).count(), 0, '最大值评估不能用均值口径说明')
        await frozen.getByText('冻结观测 · 1条', { exact: true }).click()
        await frozen.getByText('qa-frozen-file', { exact: true }).waitFor()
        assert.ok(calls.some(call => call.path.endsWith('/records/observations/qa-frozen-observation')), '原始值通过冻结报告观测接口回查')
        assert.equal(calls.filter(call => ['/api/evidence','/api/observations'].includes(call.path)).length, liveReads, '指标依据读取冻结记录，不重查当前事件或活观测')
        await frozen.screenshot({ path: resolve(artifactDir, 'evaluation-record-1440.png') })
        await details.getByRole('button', { name: '下一页', exact: true }).click()
        await details.getByText('误报样本 · qa-frozen-50', { exact: true }).waitFor()
        assert.equal(calls.findLast(call => call.path.endsWith('/metrics/mae')).query.offset, '50')
        const download = page.waitForEvent('download')
        await report.getByRole('button', { name: '下载冻结报告', exact: true }).click()
        await download
        assert.equal(calls.findLast(call => call.path.endsWith('/export')).authenticated, true)
        for (const width of [1440, 390]) {
          await page.setViewportSize({ width, height: 900 })
          await page.screenshot({ path: resolve(artifactDir, `evaluation-frozen-${width}.png`), fullPage: true })
          const overflow = await page.evaluate(() => [...document.querySelectorAll('body *')].map(element => ({ tag: element.tagName, className: String(element.className), right: element.getBoundingClientRect().right, width: element.getBoundingClientRect().width })).filter(item => item.right > innerWidth + 1 && item.width > 0).slice(0, 20))
          await writeFile(resolve(artifactDir, `evaluation-overflow-${width}.json`), JSON.stringify(overflow, null, 2))
          assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `评估页面${width}px无整页溢出`)
        }
        nextMetricGate = new Promise(resolveGate => { releaseMetric = resolveGate })
        await Promise.all([page.waitForRequest(request => request.url().includes('/metrics/precision')), report.getByRole('button', { name: '查看逐事件查准率明细', exact: true }).click()])
        await page.getByRole('combobox', { name: '数据源', exact: true }).selectOption(otherSource.source_id)
        releaseMetric()
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        assert.equal(await page.getByRole('region', { name: '指标明细', exact: true }).count(), 0, '来源切换后迟到明细不回写')
      } finally { releaseMetric(); await context.close() }
    })
    await t.test('replay speed and linked restart preserve the original session clock', async () => {
      reset('safety_management')
      const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })
      try {
        const page = await monitoredPage(context); page.setDefaultTimeout(6000)
        await page.goto(url)
        await page.locator('[name="username"]').fill('qa-only-user')
        await page.locator('[name="password"]').fill('qa-only-password-for-test')
        await page.getByRole('button', { name: /^登录工作台/ }).click()
        await page.getByRole('heading', { name: 'TEST-UI 工作面', exact: true }).waitFor()
        await page.getByRole('button', { name: '创建会话', exact: true }).click()
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        const oldId = session.session_id, oldClock = session.as_of
        await page.getByRole('combobox', { name: '回放倍速', exact: true }).selectOption('5')
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        assert.deepEqual(calls.findLast(call => call.path.endsWith('/actions')).body, { action: 'set_speed', speed: 5 })
        await page.getByRole('button', { name: '前进 60 秒 →', exact: true }).click()
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        const frozenOldClock = session.as_of
        await page.getByText('重新回放', { exact: true }).click()
        await page.getByLabel('回放起点（UTC+8）', { exact: true }).fill(mineTime(oldClock).replace(' ', 'T').slice(0,16))
        await page.getByRole('button', { name: '从此处新建回放', exact: true }).click()
        await page.waitForFunction(() => !document.querySelector('.loading-note'))
        const posted = calls.findLast(call => call.path === '/api/sessions' && call.method === 'POST').body
        assert.equal(posted.parent_session_id, oldId)
        assert.equal(posted.speed, 5)
        assert.equal(savedSessions.get(oldId).as_of, frozenOldClock)
        assert.notEqual(session.session_id, oldId)
      } finally { await context.close() }
    })
    assert.deepEqual(externalRequests, [], '浏览器仅访问本机隔离服务')
  } finally {
    await mkdir(artifactDir, { recursive: true })
    await writeFile(resolve(artifactDir, 'browser-network.json'), JSON.stringify({ origin: url, external_requests: externalRequests, model_requests: 0 }, null, 2))
    await browser.close(); await new Promise(resolveClose => server.close(resolveClose))
  }
})
