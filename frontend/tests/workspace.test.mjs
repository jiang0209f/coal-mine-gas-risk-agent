import assert from 'node:assert/strict'
import test from 'node:test'
import { briefMatches, briefScopeKey, briefCitations, briefTextParts, localCitationDetail, shouldPollBrief } from '../src/briefs.mjs'
import { createApiClient } from '../src/api.mjs'
import { adviceIsStale, canCaseAction, referenceLocation, referenceValidity } from '../src/knowledge.mjs'
import { advanceAsOf, chinaInputToIso, mineTime, observationWindow, seriesToPoints } from '../src/utils.mjs'
import { chooseModelKey, contextKey, createRequestGuard, displayWindow, forecastForEvent, forecastMatches, latestVisibleReading, modelKey, replayStart } from '../src/workspace.mjs'

test('reference locations preserve page paragraph and line anchors', () => {
  assert.equal(referenceLocation({ page: 3, paragraph: 2 }), '第3页 · 第2段')
  assert.equal(referenceLocation({ paragraph: 7, line: 12 }), '第7段 · 第12行')
  assert.equal(referenceLocation({ section: '通风管理' }), '通风管理')
  assert.equal(referenceLocation({ table: 2, row: 3 }), '表2 · 第3行')
  assert.equal(referenceLocation({ page: 4, chunk: 2, start_offset: 1100, end_offset: 2300 }), '第4页 · 片段2 · 字符1101—2300')
  assert.equal(referenceLocation({ part: 2, char_start: 0, char_end: 100 }), '片段2 · 字符1—100')
})

test('reference badges distinguish approval from current effective dates', () => {
  const now = Date.parse('2026-11-04T00:00:00Z')
  assert.equal(referenceValidity({ state: 'approved', applicability: {} }, now), 'current')
  assert.equal(referenceValidity({ state: 'approved', applicability: { valid_to: '2026-11-04T00:00:00Z' } }, now), 'expired')
  assert.equal(referenceValidity({ state: 'pending_review', applicability: { valid_from: '2026-11-05T00:00:00Z' } }, now), 'not_effective')
  assert.equal(referenceValidity({ state: 'approved', applicability: { valid_from: '2026-11-04T00:00:00Z' } }, now), 'current')
})

test('case execution requires approved current advice and filled action content', () => {
  const advice = { advice_id: 'a', version: 2, event_revision: 2 }
  const packet = { case: { phase: 'advice_reviewed' }, current_event_revision: 2,
    advice_review_status: { 'a:2': { decision: 'approved' } }, advice_validity: { 'a:2': { stale: false } } }
  const input = { packet, advice, roles: ['ventilation'], content: '已执行核实', busy: false }
  assert.equal(canCaseAction('record_execution', input), true)
  assert.equal(canCaseAction('record_execution', { ...input, content: '' }), false)
  assert.equal(canCaseAction('record_execution', { ...input, busy: true }), false)
  packet.advice_validity['a:2'].stale = true
  assert.equal(adviceIsStale(packet, advice), true)
  assert.equal(canCaseAction('record_execution', input), false)
  assert.equal(canCaseAction('review_advice', { ...input, roles: ['safety_management'] }), false)
})

test('case review respects role stage and previous decision', () => {
  const advice = { advice_id: 'a', version: 1, event_revision: 1 }
  const packet = { case: { phase: 'confirmed' }, current_event_revision: 1, advice_review_status: {} }
  const input = { packet, advice, roles: ['ventilation'], content: '' }
  assert.equal(canCaseAction('review_advice', input), false)
  assert.equal(canCaseAction('review_advice', { ...input, roles: ['safety_management'] }), true)
  packet.advice_review_status['a:1'] = { decision: 'rejected' }
  assert.equal(canCaseAction('review_advice', { ...input, roles: ['safety_management'] }), false)
  assert.equal(adviceIsStale({ current_event_revision: 2 }, advice), true)
  assert.equal(canCaseAction('review_case', { packet: { case: { phase: 'feedback_submitted' } }, roles: ['gas_governance'], content: '复核完成' }), true)
})

test('original reference downloads retain bearer auth and blob contents', async () => {
  let requested
  const api = createApiClient({ getToken: () => 'download-only-token', fetchImpl: async (url, options) => {
    requested = { url, options }
    return new Response('合成资料原文', { headers: { 'Content-Type': 'text/plain' } })
  } })
  const result = await api('/references/files/f/content', { responseType: 'blob' })
  assert.equal(requested.url, '/api/references/files/f/content')
  assert.equal(requested.options.headers.get('Authorization'), 'Bearer download-only-token')
  assert.equal(await result.text(), '合成资料原文')
})

test('API uses bearer authentication and returns the direct list', async () => {
  const calls = []
  const api = createApiClient({ getToken: () => 'test-only-token', fetchImpl: async (...args) => {
    calls.push(args)
    return new Response(JSON.stringify([{ source_id: 'test-source' }]))
  } })
  assert.deepEqual(await api('/sources'), [{ source_id: 'test-source' }])
  assert.equal(calls[0][0], '/api/sources')
  assert.equal(calls[0][1].headers.get('Authorization'), 'Bearer test-only-token')
  assert.equal(calls[0][1].body, undefined)
})

test('JSON request serializes fields and unauthorized response clears login', async () => {
  let cleared = false
  let request
  const api = createApiClient({ getToken: () => '', onUnauthorized: () => { cleared = true },
    fetchImpl: async (_, options) => {
      request = options
      return new Response(JSON.stringify({ message: '登录已过期' }), { status: 401 })
    } })
  await assert.rejects(api('/sessions', { method: 'POST', body: { source_id: 's' } }), /登录已过期/)
  assert.equal(cleared, true)
  assert.equal(request.headers.get('Content-Type'), 'application/json')
  assert.deepEqual(JSON.parse(request.body), { source_id: 's' })
})

test('multipart upload preserves the browser-generated boundary', async () => {
  let request
  const form = new FormData()
  form.append('file', new Blob(['data']), 'test.csv')
  const api = createApiClient({ fetchImpl: async (_, options) => {
    request = options
    return new Response(JSON.stringify({ file_id: 'test-file' }))
  } })
  assert.equal((await api('/files', { method: 'POST', body: form })).file_id, 'test-file')
  assert.equal(request.headers.has('Content-Type'), false)
  assert.equal(request.body, form)
})

test('validation errors retain the server field message', async () => {
  const api = createApiClient({ fetchImpl: async () => new Response(JSON.stringify({
    detail: [{ loc: ['body', 'source_id'], msg: 'Field required' }],
  }), { status: 422 }) })
  await assert.rejects(api('/sessions'), /source_id.*Field required/)
})

test('forward playback advances exactly sixty seconds across midnight', () => {
  assert.equal(advanceAsOf('2022-01-01T23:59:30+08:00'), '2022-01-01T16:00:30.000Z')
  assert.throws(() => advanceAsOf('2022-01-01 00:00:00'), /时区/)
})

test('mine time inputs use UTC+8 independent of the browser time zone', () => {
  assert.equal(chinaInputToIso('2022-01-01T00:00'), '2021-12-31T16:00:00.000Z')
  assert.throws(() => chinaInputToIso(''), /时间/)
})

test('chart timestamps render in mine time from numeric epoch values', () => {
  assert.match(mineTime(Date.parse('2022-01-01T00:00:00+08:00')), /2022-01-01 00:00:00/)
})

test('observation window ends at the current as_of and retains real gaps', () => {
  const asOf = '2022-01-01T01:00:00+08:00'
  const window = observationWindow(asOf, 60)
  assert.equal(window.end_at, '2021-12-31T17:00:00.000Z')
  assert.equal(window.start_at, '2021-12-31T16:00:00.000Z')
  const packet = { rows: [
    { event_time: '2022-01-01T00:58:00+08:00', value: 0.1 },
    { event_time: '2022-01-01T00:59:00+08:00', value: null },
    { event_time: '2022-01-01T01:01:00+08:00', value: 0.9 },
  ] }
  assert.deepEqual(seriesToPoints(packet, asOf), [
    [Date.parse('2022-01-01T00:58:00+08:00'), 0.1],
    [Date.parse('2022-01-01T00:59:00+08:00'), null],
  ])
})

test('model selection preserves an explicit version and falls back to a default model', () => {
  const baseline = { model_id: 'base', version: 'v1', default_eligible: true, created_at: '2022-01-01T00:00:00Z' }
  const professional = { model_id: 'trained', version: 'v2', default_eligible: false }
  assert.equal(chooseModelKey([professional, baseline]), 'base:v1')
  assert.equal(chooseModelKey([professional, baseline], 'trained:v2'), 'trained:v2')
  assert.equal(chooseModelKey([baseline], 'trained:v2'), 'base:v1')
  assert.equal(modelKey(professional), 'trained:v2')
  assert.equal(chooseModelKey([]), '')
})

test('prediction restore rejects another session source point version and future run', () => {
  const session = { session_id: 's', source_id: 'source', as_of: '2022-02-05T13:12:00Z' }
  const run = { session_id: 's', source_id: 'source', point_id: 'T0', model_id: 'trained', model_version: 'v2', as_of: '2022-02-05T13:10:00Z', cutoff: '2022-02-05T13:10:00Z' }
  assert.equal(forecastMatches(run, session, 'T0', 'trained:v2'), true)
  for (const update of [{ session_id: 'other' }, { source_id: 'other' }, { point_id: 'wind' }, { model_version: 'v1' }, { as_of: '2022-02-05T13:13:00Z' }, { cutoff: '2022-02-05T13:13:00Z' }]) {
    assert.equal(forecastMatches({ ...run, ...update }, session, 'T0', 'trained:v2'), false)
  }
  assert.equal(forecastMatches(null, session, 'T0', 'trained:v2'), false)
})

test('superseded responses are ignored even when the selected point returns to its previous value', () => {
  const guard = createRequestGuard()
  const first = guard.begin()
  guard.invalidate()
  const next = guard.begin()
  assert.equal(guard.isCurrent(first), false)
  assert.equal(guard.isCurrent(next), true)
  guard.invalidate()
  assert.equal(guard.isCurrent(next), false)
  const session = { session_id: 's', source_id: 'source', as_of: '2022-02-05T13:12:00Z' }
  assert.notEqual(contextKey(session, 'T0'), contextKey({ ...session, as_of: '2022-02-05T13:13:00Z' }, 'T0'))
  assert.notEqual(contextKey(session, 'T0'), contextKey(session, 'wind'))
})

test('event focus loads its historical neighbourhood instead of the current hour', () => {
  const asOf = '2022-02-05T20:00:00Z'
  const event = { start_at: '2022-02-05T13:07:00Z', event_time: '2022-02-05T13:12:00Z' }
  const window = displayWindow(asOf, event)
  assert.equal(window.start_at, '2022-02-05T12:07:00.000Z')
  assert.equal(window.end_at, '2022-02-05T13:22:00.000Z')
  assert.equal(displayWindow('2022-02-05T13:12:00Z', event).end_at, '2022-02-05T13:12:00.000Z')
  assert.deepEqual(displayWindow(asOf), observationWindow(asOf))
})

test('new replay starts from actual coverage and shows the first closed minute', () => {
  assert.equal(replayStart({ start_at: '2022-02-05T13:07:00Z', end_at: '2022-02-05T14:00:00Z' }), '2022-02-05T21:08:00')
  assert.equal(replayStart({ start_at: null, end_at: null }), '')
  assert.equal(replayStart({ start_at: '2022-02-05T13:07:00Z', end_at: '2022-02-05T13:07:00Z' }), '2022-02-05T21:07:00')
})

test('observed event retains compatible prediction while forecast event displays its linked run', () => {
  const session = { session_id: 's', source_id: 'source', as_of: '2022-02-05T13:12:00Z' }
  const current = { run_id: 'current', session_id: 's', source_id: 'source', point_id: 'T0', model_id: 'trained', model_version: 'v2', as_of: '2022-02-05T13:10:00Z' }
  assert.equal(forecastForEvent(current, { forecast_run: null }, session, 'T0', 'trained:v2'), current)
  const linked = { ...current, run_id: 'linked', model_id: 'baseline', model_version: 'v1' }
  assert.equal(forecastForEvent(current, { forecast_run: linked }, session, 'T0', 'trained:v2'), linked)
  assert.equal(forecastForEvent(current, { forecast_run: { ...linked, source_id: 'other' } }, session, 'T0', 'trained:v2'), current)
  assert.equal(forecastForEvent({ ...current, point_id: 'wind' }, {}, session, 'T0', 'trained:v2'), null)
})

test('channel cards use the latest visible valid row and reject future available times', () => {
  const asOf = '2022-02-05T13:12:00Z'
  const visible = { event_time: '2022-02-05T13:10:00Z', available_at: '2022-02-05T13:10:00Z', bucket_end: '2022-02-05T13:11:00Z', value: 0.7 }
  const packet = { rows: [
    { event_time: '2022-02-05T13:13:00Z', value: 9.99 },
    { ...visible, event_time: '2022-02-05T13:11:00Z', available_at: '2022-02-05T13:13:00Z', value: 8.88 },
    { ...visible, event_time: '2022-02-05T13:12:00Z', bucket_end: '2022-02-05T13:13:00Z', value: 7.77 },
    { ...visible, event_time: '2022-02-05T13:11:00Z', value: 6.66, quality_flags: ['invalid_value'] },
    visible,
    { ...visible, event_time: '2022-02-05T13:09:00Z', value: 0.6 },
  ] }
  assert.deepEqual(latestVisibleReading(packet, asOf), visible)
  assert.equal(latestVisibleReading({ rows: [{ event_time: '2022-02-05T13:13:00Z', value: 9.99 }] }, asOf), null)
})

test('event brief matches only its visible session source event and revision', () => {
  const session = { session_id: 's', source_id: 'source', as_of: '2022-02-05T13:12:00Z' }
  const event = { event_id: 'e', event_revision: 2, point_id: 'T0' }
  const brief = { session_id: 's', source_id: 'source', event_id: 'e', event_revision: 2, as_of: '2022-02-05T13:10:00Z', status: 'running' }
  assert.equal(briefMatches(brief, session, event), true)
  for (const update of [{ session_id: 'other' }, { source_id: 'other' }, { event_id: 'other' }, { event_revision: 1 }, { as_of: '2022-02-05T13:13:00Z' }]) assert.equal(briefMatches({ ...brief, ...update }, session, event), false)
  assert.equal(briefMatches(null, session, event), false)
  assert.equal(briefMatches({ ...brief, event_revision: 1, requested_event_revision: 2, stale: true }, session, event), true)
  assert.equal(briefMatches({ ...brief, event_revision: 1, requested_event_revision: 3, stale: true }, session, event), false)
  assert.equal(briefMatches({ ...brief, event_revision: 3, requested_event_revision: 2, stale: true }, session, event), false)
  assert.equal(briefMatches({ ...brief, task: { session_id: 'other', source_id: 'source', event_id: 'e', event_revision: 2 } }, session, event), false)
  assert.notEqual(briefScopeKey(session, event), briefScopeKey(session, { ...event, event_revision: 3 }))
  assert.notEqual(briefScopeKey(session, event), briefScopeKey({ ...session, as_of: '2022-02-05T13:13:00Z' }, event))
})

test('brief polling is bounded to queued and running states', () => {
  for (const status of ['queued', 'running']) assert.equal(shouldPollBrief({ status }), true)
  for (const status of ['not_started', 'completed', 'partial', 'failed', 'cancelled']) assert.equal(shouldPollBrief({ status }), false)
  assert.equal(shouldPollBrief(null), false)
})

test('only returned citations become links and unknown model markers stay text', () => {
  const citation = { citation_id: 'E1', kind: 'event', label: '预警事件', event_id: 'e', event_revision: 2 }
  const packet = { task: { citations: [citation, { ...citation }, { citation_id: 'E2', kind: 'external_url', href: 'https://untrusted.invalid' }] } }
  assert.deepEqual(briefCitations(packet), [citation])
  assert.deepEqual(briefTextParts('依据[E1]，未知[E99]。', briefCitations(packet)), [
    { text: '依据' }, { text: '[E1]', citation }, { text: '，未知[E99]。' },
  ])
})

test('brief references open the frozen task cards and retain their original revision', () => {
  const old = { event_id: 'e', event_revision: 1, point_id: 'T0', start_at: '2022-02-05T13:07:00Z' }
  const observation = { id: 'old-observation', event_time: '2022-02-05T13:07:00Z', value: .7, source_file_id: 'original-file', row_number: 8 }
  const frozen = { task: { result_cards: [{ tool: 'get_event_evidence', data: { event_snapshot: old, observation_fragments: [observation], rules: { rule_version: 'rule-v1' } } }] } }
  const eventDetail = localCitationDetail(frozen, { kind: 'event', event_id: 'e', event_revision: 1, label: '事件' })
  assert.equal(eventDetail.event, old)
  assert.equal(localCitationDetail(frozen, { kind: 'observation', observation_id: 'old-observation' }).observations[0], observation)
  assert.equal(localCitationDetail(frozen, { kind: 'rule' }).rules.rule_version, 'rule-v1')
  assert.equal(localCitationDetail(frozen, { kind: 'observation', observation_id: 'not-in-frozen-card' }), null)
})

test('brief display removes heading and emphasis markers while keeping original text and citations', () => {
  const citation = { citation_id: 'E1', kind: 'event', event_id: 'e', event_revision: 1 }
  const task = { final_text: '## **事件概况**\n**实测预警** [E1]\n### 待核实\n__监测记录__。' }
  const original = task.final_text
  assert.deepEqual(briefTextParts(task.final_text, [citation]), [
    { text: '事件概况\n实测预警 ' }, { text: '[E1]', citation }, { text: '\n待核实\n监测记录。' },
  ])
  assert.equal(task.final_text, original)
})

test('brief display translates business codes but preserves identifiers timestamps and unmatched syntax', () => {
  const source = '事件 observed_exceedance；forecast_exceedance。质量：availability_inferred、unit_inferred、suspected_calibration、planned_calibration、confirmed_calibration。\nmodel_id=fedformer-lgbm-31002 version=trained-20261103-v1 observed_exceedance-v2 #1测点\nmodel_id=unit_inferred version: "observed_exceedance"\n2022-02-05T13:12:00+00:00，13:07，2 ** 3；<img src=x>'
  const text = briefTextParts(source, []).map(part => part.text).join('')
  assert.ok(text.includes('事件 实测超限；预测预警。质量：可用时间推定、单位推定、疑似标校、计划标校、已确认标校。'))
  for (const literal of ['model_id=fedformer-lgbm-31002', 'version=trained-20261103-v1', 'model_id=unit_inferred', 'version: "observed_exceedance"', 'observed_exceedance-v2', '#1测点', '2022-02-05T13:12:00+00:00', '13:07', '2 ** 3', '<img src=x>']) assert.ok(text.includes(literal), literal)
})
