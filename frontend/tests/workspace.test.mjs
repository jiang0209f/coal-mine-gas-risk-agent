import assert from 'node:assert/strict'
import test from 'node:test'
import { createApiClient } from '../src/api.mjs'
import { advanceAsOf, chinaInputToIso, mineTime, observationWindow, seriesToPoints } from '../src/utils.mjs'

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
