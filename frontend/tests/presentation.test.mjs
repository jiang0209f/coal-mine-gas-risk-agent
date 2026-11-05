import assert from 'node:assert/strict'
import test from 'node:test'
import { createApiClient } from '../src/api.mjs'

const aliases = [['12345', 'FACE-A'], ['某某煤矿', '示范矿井']]
const jsonResponse = value => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } })

test('anonymous responses cover names, identifiers, dictionary keys and original passages without changing readings', async () => {
  const original = { source_id: 'multi-12345', face_id: '12345', title: '某某煤矿12345工作面规程',
    text: '12345工作面第7页', prediction_models: { '12345::T0': { model_id: 'forecast-12345-v1' } },
    raw_value: '12345', value: 12345, records_sha256: 'a'.repeat(30) + '12345' + 'b'.repeat(29) }
  const api = createApiClient({ presentation: true, aliases, fetchImpl: async () => jsonResponse(original) })
  const value = await api('/sources')
  assert.equal(value.source_id, 'multi-FACE-A')
  assert.equal(value.face_id, 'FACE-A')
  assert.equal(value.title, '示范矿井FACE-A工作面规程')
  assert.equal(value.text, 'FACE-A工作面第7页')
  assert.equal(value.prediction_models['FACE-A::T0'].model_id, 'forecast-FACE-A-v1')
  assert.equal(value.raw_value, '12345')
  assert.equal(value.value, 12345)
  assert.equal(value.records_sha256, original.records_sha256)
  assert.equal(original.face_id, '12345')
})

test('anonymous requests restore source and point identities in encoded URLs and JSON without mutating state', async () => {
  const body = { source_id: 'multi-FACE-A', prediction_models: { 'FACE-A::T0': 'forecast-FACE-A-v1' } }
  let seen
  const api = createApiClient({ presentation: true, aliases, fetchImpl: async (url, options) => {
    seen = { url, body: JSON.parse(options.body) }; return jsonResponse({ ok: true })
  } })
  await api('/points/FACE-A%3A%3AT0?source_id=multi-FACE-A', { method: 'POST', body })
  assert.equal(seen.url, '/api/points/12345%3A%3AT0?source_id=multi-12345')
  assert.equal(seen.body.source_id, 'multi-12345')
  assert.equal(seen.body.prediction_models['12345::T0'], 'forecast-12345-v1')
  assert.equal(body.source_id, 'multi-FACE-A')
})

test('credentials and measured raw strings are not rewritten while user scope is aliased', async () => {
  let sent
  const api = createApiClient({ presentation: true, aliases, fetchImpl: async (_, options) => {
    sent = JSON.parse(options.body)
    return jsonResponse({ access_token: 'credential-12345-FACE-A', user: { face_ids: ['12345'] } })
  } })
  const result = await api('/auth/login', { method: 'POST', body: { username: 'operator-FACE-A', password: 'secret-FACE-A' } })
  assert.equal(sent.username, 'operator-FACE-A')
  assert.equal(sent.password, 'secret-FACE-A')
  assert.equal(result.access_token, 'credential-12345-FACE-A')
  assert.deepEqual(result.user.face_ids, ['FACE-A'])
})

test('multipart fields restore aliases and preserve the file object and bytes', async () => {
  const body = new FormData()
  body.append('face_ids', '["FACE-A"]')
  body.append('file', new Blob(['original FACE-A bytes']), 'sample.txt')
  let received
  const api = createApiClient({ presentation: true, aliases, fetchImpl: async (_, options) => { received = options.body; return jsonResponse({ file_id: 'file' }) } })
  await api('/references/files', { method: 'POST', body })
  assert.equal(received.get('face_ids'), '["12345"]')
  assert.equal(await received.get('file').text(), 'original FACE-A bytes')
  assert.equal(body.get('face_ids'), '["FACE-A"]')
})

test('anonymous mode refuses binary originals before requesting their content', async () => {
  let calls = 0
  const api = createApiClient({ presentation: true, aliases, fetchImpl: async () => { calls++; return new Response('original') } })
  await assert.rejects(api('/references/files/file/content', { responseType: 'blob' }), /原件/)
  assert.equal(calls, 0)
})

test('anonymous mode downloads only the presentation investigation DOCX and preserves session scope', async () => {
  let seen
  const api = createApiClient({ presentation: true, aliases, fetchImpl: async url => {
    seen = url
    return new Response('presentation-docx', { headers: { 'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' } })
  } })
  const blob = await api('/investigations/report-id/report.docx?session_id=scope-FACE-A&presentation=false', { responseType: 'blob' })
  assert.equal(await blob.text(), 'presentation-docx')
  assert.equal(seen, '/api/investigations/report-id/report.docx?session_id=scope-12345&presentation=true')
  for (const path of ['/investigations/report-id/report.docx.extra', '/investigations/report-id/report.pdf', '/references/files/file/content']) {
    await assert.rejects(api(path, { responseType: 'blob' }), /原件/)
  }
})

test('anonymous report download has an explicit display projection and its own verifiable digest', async () => {
  const original = { report_sha256: 'original-frozen-digest', projection_sha256: 'original-export-digest', report: { face_id: '12345', text: '某某煤矿12345工作面' } }
  const api = createApiClient({ presentation: true, aliases, fetchImpl: async () => jsonResponse(original) })
  const blob = await api('/evaluations/report/export', { responseType: 'blob' })
  const output = JSON.parse(await blob.text())
  assert.equal(output.format, 'anonymous-display-v1')
  assert.equal(output.source_report_sha256, 'original-frozen-digest')
  assert.equal(output.records.report.face_id, 'FACE-A')
  assert.equal(output.records.report.text, '示范矿井FACE-A工作面')
  assert.equal(output.records.projection_sha256, undefined)
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(output.records)))
  assert.equal(output.display_sha256, Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join(''))
  assert.doesNotMatch(await blob.text(), /12345|某某煤矿/)
})

test('server error messages are aliased and standard mode keeps response and binary contracts', async () => {
  const anonymous = createApiClient({ presentation: true, aliases, fetchImpl: async () => new Response(JSON.stringify({ message: '12345工作面未就绪' }), { status: 409 }) })
  await assert.rejects(anonymous('/sources'), /FACE-A工作面未就绪/)
  const standard = createApiClient({ presentation: false, aliases, fetchImpl: async () => jsonResponse({ face_id: '12345' }) })
  assert.deepEqual(await standard('/sources'), { face_id: '12345' })
  const binary = createApiClient({ presentation: false, fetchImpl: async () => new Response('original-binary') })
  assert.equal(await (await binary('/file', { responseType: 'blob' })).text(), 'original-binary')
})

test('real report and manifest export shape identifies original digests as source identities', async () => {
  const original = {
    report: { face_id: '12345', report_sha256: 'frozen-report', records_sha256: 'frozen-records', manifest_sha256: 'frozen-manifest' },
    manifest: { face_id: '12345', frozen_records: { observations: [{ point_id: '12345::T0', raw_value: '0.67' }] },
      records_sha256: 'frozen-records', export_records_sha256: 'original-projection', export_manifest_sha256: 'original-export' },
  }
  const api = createApiClient({ presentation: true, aliases, fetchImpl: async () => jsonResponse(original) })
  const output = JSON.parse(await (await api('/evaluations/report/export', { responseType: 'blob' })).text())
  assert.equal(output.records.manifest.export_records_sha256, undefined)
  assert.equal(output.records.manifest.export_manifest_sha256, undefined)
  assert.equal(output.records.manifest.source_export_records_sha256, 'original-projection')
  assert.equal(output.records.manifest.source_export_manifest_sha256, 'original-export')
  assert.equal(output.records.report.report_sha256, undefined)
  assert.equal(output.records.report.source_report_sha256, 'frozen-report')
  assert.equal(output.records.report.source_records_sha256, 'frozen-records')
  assert.equal(output.records.manifest.frozen_records.observations[0].point_id, 'FACE-A::T0')
})

test('alias configuration rejects ambiguity, replacement chains and an empty anonymous setup', async () => {
  const { createPresentationPolicy } = await import('../src/presentation.mjs')
  assert.throws(() => createPresentationPolicy([]), /别名/)
  assert.throws(() => createPresentationPolicy([['12345', 'A'], ['45678', 'A']]), /别名/)
  assert.throws(() => createPresentationPolicy([['12345', 'FACE-A'], ['FACE-A', 'FACE-B']]), /别名/)
})
