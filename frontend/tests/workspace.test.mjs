import assert from 'node:assert/strict'
import test from 'node:test'
import { createApiClient } from '../src/api.mjs'
import { advanceAsOf, chinaInputToIso, mineTime, observationWindow, seriesToPoints } from '../src/utils.mjs'
import { chooseModelKey, contextKey, createRequestGuard, displayWindow, forecastForEvent, forecastMatches, latestVisibleReading, modelKey, replayStart } from '../src/workspace.mjs'

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
