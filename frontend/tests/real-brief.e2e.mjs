// Manual local acceptance: real historical observations and server-side DeepSeek calls.
// Credentials arrive through stdin and are never written to the report.
import assert from 'node:assert/strict'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { chromium } from 'playwright'
import { mineTime } from '../src/utils.mjs'

const input = []
for await (const chunk of process.stdin) input.push(chunk)
const config = JSON.parse(Buffer.concat(input).toString('utf8'))
const origin = new URL(config.baseUrl).origin
assert.equal(new URL(config.baseUrl).hostname, '127.0.0.1')
assert.ok(config.username && config.password && config.session_id && config.outputDir && config.counterPath)
const output = resolve(config.outputDir)
await mkdir(output, { recursive: true })
const browserPath = process.env.UI_TEST_BROWSER || [
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
].find(path => existsSync(path))
assert.ok(browserPath, 'An installed Edge or Chrome is required')
const browser = await chromium.launch({ executablePath: browserPath, headless: true })
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
const page = await context.newPage()
page.setDefaultTimeout(15000)
const report = { status: 'running', source_id: config.sourceId, session_id: config.session_id,
  data_scope: config.dataScope || 'real_historical', model_mode: config.modelMode || 'deepseek',
  checks: [], observed_states: [], viewports: [], page_errors: [], browser_external_requests: [], generated_by_manual_post: false }
let accessToken
function safeMessage(value) {
  let text = String(value)
  for (const secret of [config.username, config.password, accessToken]) if (secret) text = text.replaceAll(secret, '[redacted]')
  return text.replace(/Bearer\s+[^\s"']+/gi, 'Bearer [redacted]')
}
page.on('pageerror', error => report.page_errors.push(safeMessage(error.message)))
const briefPosts = []
page.on('request', request => {
  const url = new URL(request.url())
  if (/\/brief$/.test(url.pathname) && request.method() === 'POST') briefPosts.push(url.pathname)
})
await context.route('**/*', async route => {
  const url = new URL(route.request().url())
  if (url.origin === origin || url.protocol === 'data:') await route.continue()
  else { report.browser_external_requests.push(url.origin); await route.abort() }
})
const mask = () => [page.locator('.user-chip'), page.locator('[name="username"]'), page.locator('[name="password"]')]
const headers = () => ({ Authorization: `Bearer ${accessToken}` })
async function get(path) {
  const response = await context.request.get(config.baseUrl + path, { headers: headers() })
  assert.equal(response.status(), 200, path)
  return response.json()
}
async function waitReady() {
  await page.waitForFunction(() => !document.querySelector('.loading-note'))
}
async function modelRequestCount() {
  return JSON.parse(await readFile(config.counterPath, 'utf8')).chat_requests
}
function observe(state) {
  if (state && !report.observed_states.includes(state)) report.observed_states.push(state)
}

try {
  await page.goto(config.baseUrl)
  await page.locator('[name="username"]').fill(config.username)
  await page.locator('[name="password"]').fill(config.password)
  const loggedIn = page.waitForResponse(response => new URL(response.url()).pathname === '/api/auth/login')
  await page.getByRole('button', { name: /^登录工作台/ }).click()
  accessToken = (await (await loggedIn).json()).access_token
  await page.getByRole('region', { name: '综合态势', exact: true }).waitFor()
  await waitReady()
  await page.getByRole('combobox', { name: '数据源', exact: true }).selectOption(config.sourceId)
  await waitReady()
  await page.getByRole('combobox', { name: '已有会话', exact: true }).selectOption(config.session_id)
  await waitReady()
  assert.equal(await modelRequestCount(), 0)
  let triggered
  for (let minute = 7; minute <= 12; minute++) {
    const tickResponse = page.waitForResponse(response => new URL(response.url()).pathname === `/api/sessions/${config.session_id}/tick`)
    await page.getByRole('button', { name: '前进 60 秒 →', exact: true }).click()
    const response = await tickResponse
    assert.equal(response.status(), 200)
    const tick = await response.json()
    assert.equal(Date.parse(tick.session.as_of), Date.parse(`2022-02-05T13:${String(minute).padStart(2, '0')}:00+00:00`))
    await waitReady()
    if (minute < 12) assert.equal(tick.briefs.length, 0)
    else triggered = tick
  }
  assert.equal(triggered.briefs.length, 1)
  const reference = triggered.briefs[0]
  observe(reference.status)
  report.trigger = reference
  report.checks.push('Six UI replay ticks create a new observed event and automatically queue its brief')
  const params = new URLSearchParams({ session_id: config.session_id, event_revision: String(reference.event_revision) })
  const briefPath = `/api/events/${reference.event_id}/brief?${params}`
  const region = page.getByRole('region', { name: '研判简报', exact: true })
  await region.waitFor()
  let brief
  for (let count = 0; count < 360; count++) {
    brief = await get(briefPath)
    observe(brief.status)
    if (!['queued', 'running'].includes(brief.status)) break
    await new Promise(resolveWait => setTimeout(resolveWait, 250))
  }
  report.brief_status = brief.status
  report.task = { task_id: brief.task_id, event_id: brief.event_id, event_revision: brief.event_revision,
    as_of: brief.as_of, model: brief.task?.model, method: brief.task?.method, usage: brief.task?.usage,
    document_search_state: brief.task?.document_search_state,
    error_code: brief.task?.error_code, upstream_status_code: brief.task?.upstream_status_code,
    tool_steps: brief.task?.tool_steps, citations: brief.task?.citations,
    cited_ids: brief.task?.cited_ids, final_text: brief.task?.final_text }
  assert.ok(['completed', 'partial'].includes(brief.status), `Real brief ended in ${brief.status}: ${brief.task?.error_code || ''}`)
  assert.ok(brief.task.model && brief.task.method === 'deepseek')
  assert.ok(brief.task.final_text && brief.task.cited_ids?.length)
  assert.ok(brief.task.tool_steps.some(step => step.name === 'get_event_evidence' && step.status === 'success'))
  const citationIds = new Set(brief.task.citations.map(citation => citation.citation_id))
  assert.ok(brief.task.cited_ids.every(id => citationIds.has(id)))
  report.body_format = { markdown_markers: /\*\*|^#{1,6}\s/m.test(brief.task.final_text),
    internal_event_quality_codes: /observed_exceedance|forecast_exceedance|availability_inferred|unit_inferred/.test(brief.task.final_text),
    mentions_mine_event_time: /21:07|21时0?7/.test(brief.task.final_text), document_search_state: brief.task.document_search_state }
  await region.getByText(brief.status === 'completed' ? '已完成' : '部分完成', { exact: true }).first().waitFor()
  await region.getByRole('region', { name: '工具调用过程', exact: true }).getByText(/读取事件依据/).waitFor()
  report.checks.push(`${report.model_mode === 'offline' ? 'Offline injected adapter' : 'DeepSeek'} reply retains a completed evidence tool step and valid citation identifiers`)
  const evidenceCard = brief.task.result_cards.find(card => card.tool === 'get_event_evidence')?.data
  if (config.expectedAttribution) {
    assert.equal(evidenceCard?.candidate_attribution?.status, 'completed')
    assert.equal(evidenceCard.candidate_attribution.input_refs.length, 366)
    report.attribution = { status: evidenceCard.candidate_attribution.status, model_id: evidenceCard.candidate_attribution.model_id,
      model_version: evidenceCard.candidate_attribution.model_version, input_reference_count: 366 }
  }
  await page.getByRole('region', { name: '工具调用过程', exact: true }).screenshot({ path: resolve(output, 'real-brief-tools.png') })

  const eventCitation = brief.task.citations.find(citation => citation.kind === 'event')
  await page.getByRole('button', { name: `打开依据 ${eventCitation.citation_id}`, exact: true }).click()
  const detail = page.getByRole('region', { name: '引用依据', exact: true })
  await detail.getByText(reference.event_id, { exact: true }).waitFor()
  assert.ok((await detail.innerText()).includes(`第 ${reference.event_revision} 版`))
  await detail.screenshot({ path: resolve(output, 'real-brief-reference.png') })
  await page.getByRole('button', { name: '关闭依据', exact: true }).click()
  const observationCitation = brief.task.citations.find(citation => citation.kind === 'observation')
  if (observationCitation) {
    await page.getByRole('button', { name: `打开依据 ${observationCitation.citation_id}`, exact: true }).click()
    await detail.getByRole('columnheader', { name: '原始行', exact: true }).waitFor()
    assert.ok((await detail.innerText()).includes(String(observationCitation.row_number)))
    await page.getByRole('button', { name: '关闭依据', exact: true }).click()
  }
  report.checks.push('Event and raw-observation citations open their frozen revision and original row')

  const countBeforeReuse = await modelRequestCount()
  const duplicate = await context.request.post(`${config.baseUrl}/api/sessions/${config.session_id}/tick`, {
    headers: headers(), data: { target_as_of: triggered.session.as_of },
  })
  assert.equal(duplicate.status(), 200)
  assert.equal((await duplicate.json()).reused, true)
  await new Promise(resolveWait => setTimeout(resolveWait, 600))
  assert.equal(await modelRequestCount(), countBeforeReuse)
  const nextResponse = page.waitForResponse(response => new URL(response.url()).pathname === `/api/sessions/${config.session_id}/tick`)
  await page.getByRole('button', { name: '前进 60 秒 →', exact: true }).click()
  const next = await (await nextResponse).json()
  assert.equal(next.briefs.length, 0)
  await waitReady()
  const currentEvent = next.events.find(event => event.event_id === reference.event_id)
  assert.ok(currentEvent.event_revision > reference.event_revision)
  const previous = await get(`/api/events/${reference.event_id}/brief?${new URLSearchParams({ session_id: config.session_id, event_revision: String(currentEvent.event_revision) })}`)
  assert.equal(previous.task_id, brief.task_id)
  assert.equal(previous.stale, true)
  assert.equal(previous.event_revision, reference.event_revision)
  await new Promise(resolveWait => setTimeout(resolveWait, 600))
  assert.equal(await modelRequestCount(), countBeforeReuse)
  report.model_chat_requests = countBeforeReuse
  report.checks.push('Duplicate tick and ordinary revision reuse the frozen brief without another model request')
  assert.deepEqual(briefPosts, [], 'Automatic generation must not be replaced by a manual POST')

  for (const viewport of [{ width: 1920, height: 1080 }, { width: 1440, height: 900 }, { width: 1280, height: 800 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport)
    await page.evaluate(() => document.fonts.ready)
    const dimensions = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }))
    assert.ok(dimensions.page <= dimensions.viewport + 1, `Horizontal overflow at ${viewport.width}px`)
    await page.getByRole('button', { name: `打开依据 ${eventCitation.citation_id}`, exact: true }).click()
    await detail.getByText(reference.event_id, { exact: true }).waitFor()
    await page.getByRole('button', { name: '关闭依据', exact: true }).click()
    await page.evaluate(() => { const body = document.querySelector('.investigation-body'); if (body) body.scrollTop = 0; scrollTo(0, 0) })
    await page.screenshot({ path: resolve(output, `real-brief-${viewport.width}.png`), fullPage: true, mask: mask(), maskColor: '#0b1e30' })
    report.viewports.push({ ...viewport, ...dimensions, horizontal_overflow: false, citations_accessible: true })
  }
  report.checks.push('Brief, tools and citation controls remain accessible at 1920/1440/1280/390 pixels')
  await page.getByRole('combobox', { name: '监测点', exact: true }).selectOption(config.windPointId)
  await waitReady()
  assert.equal(await page.getByRole('region', { name: '研判简报', exact: true }).count(), 0)
  report.checks.push('Changing the selected point clears the old event brief')
  assert.deepEqual(report.page_errors, [])
  assert.deepEqual(report.browser_external_requests, [])
  report.status = 'passed'
} catch (error) {
  report.status = 'failed'
  report.error = safeMessage(error.message)
  await page.screenshot({ path: resolve(output, 'real-brief-failure.png'), fullPage: true, mask: mask(), maskColor: '#0b1e30' }).catch(() => {})
  process.exitCode = 1
} finally {
  await writeFile(resolve(output, 'browser-report.json'), JSON.stringify(report, null, 2))
  await context.close()
  await browser.close()
}
console.log(JSON.stringify({ status: report.status, checks: report.checks.length, report: resolve(output, 'browser-report.json') }))
