// Manual acceptance against an isolated local runtime with real historical observations.
// Credentials and session selection arrive through stdin; this script is not part of default CI.
import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { chromium } from 'playwright'
import { mineTime } from '../src/utils.mjs'

const chunks = []
for await (const chunk of process.stdin) chunks.push(chunk)
const config = JSON.parse(Buffer.concat(chunks).toString('utf8'))
assert.equal(new URL(config.baseUrl).hostname, '127.0.0.1', 'Acceptance requires a loopback server')
assert.ok(config.username && config.password && config.session_id && config.outputDir)
const output = resolve(config.outputDir)
await mkdir(output, { recursive: true })
const browserPath = process.env.UI_TEST_BROWSER || [
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
].find(path => existsSync(path))
assert.ok(browserPath, 'An installed Edge or Chrome is required')
const browser = await chromium.launch({ executablePath: browserPath, headless: true })
const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } })
const foreignRequests = [], pageErrors = [], requests = []
const report = { status: 'running', source_id: config.sourceId, point_id: config.pointId, session_id: config.session_id, checks: [], viewports: [] }
const page = await context.newPage()
page.setDefaultTimeout(15000)
page.on('pageerror', error => pageErrors.push(safeMessage(error.message)))
page.on('request', request => {
  const url = new URL(request.url())
  if (url.pathname.startsWith('/api/') && !url.pathname.startsWith('/api/auth/')) requests.push({ path: url.pathname, method: request.method(), query: Object.fromEntries(url.searchParams) })
})
await context.route('**/*', async route => {
  const url = new URL(route.request().url())
  if (url.origin === new URL(config.baseUrl).origin || url.protocol === 'data:') await route.continue()
  else { foreignRequests.push(url.origin); await route.abort() }
})
const screenshotMask = () => [page.locator('.user-chip'), page.locator('[name="username"]'), page.locator('[name="password"]')]
let accessToken
function safeMessage(message) {
  let result = String(message)
  for (const value of [config.username, config.password, accessToken]) if (value) result = result.replaceAll(value, '[redacted]')
  return result.replace(/Bearer\s+[^\s"']+/gi, 'Bearer [redacted]')
}
const headers = () => ({ Authorization: `Bearer ${accessToken}` })
const api = async (path) => {
  const response = await context.request.get(config.baseUrl + path, { headers: headers() })
  assert.equal(response.status(), 200, `Oracle request ${path}`)
  return response.json()
}
async function waitReady() {
  await page.waitForFunction(() => !document.querySelector('.loading-note'))
}
async function checkForecast(run) {
  assert.equal(run.status, 'completed')
  const output = run.outputs.find(item => item.quantity_semantics === 'physical_forecast' && item.prediction_stage === 'residual_corrected')
    || run.outputs.find(item => item.quantity_semantics === 'physical_forecast')
  assert.ok(output)
  assert.equal(output.interval_values.length, 12)
  const trend = page.getByRole('region', { name: '监测曲线', exact: true })
  const details = trend.locator('details.forecast-details')
  if ((await details.getAttribute('open')) === null) await details.locator('summary').click()
  const table = page.getByRole('table', { name: '预测区间数据', exact: true })
  const rows = table.locator('tbody tr')
  assert.equal(await rows.count(), output.interval_values.length)
  for (let index = 0; index < output.interval_values.length; index++) {
    const interval = output.interval_values[index]
    const text = await rows.nth(index).innerText()
    assert.ok(text.includes(mineTime(interval.start_at)), 'Interval start matches model response')
    assert.ok(text.includes(mineTime(interval.end_at)), 'Interval end matches model response')
    assert.ok(text.includes(interval.value.toFixed(3)), 'Interval value matches model response')
  }
  assert.ok((await trend.innerText()).includes(run.model_version))
  await details.locator('summary').click()
  return { run_id: run.run_id, model_id: run.model_id, model_version: run.model_version, status: run.status,
    as_of: run.as_of, cutoff: run.cutoff, interval_count: output.interval_values.length, input_reference_count: run.input_refs.length }
}

try {
  await page.goto(config.baseUrl)
  await page.locator('[name="username"]').fill(config.username)
  await page.locator('[name="password"]').fill(config.password)
  const loginResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/auth/login')
  await page.getByRole('button', { name: /^登录工作台/ }).click()
  accessToken = (await (await loginResponse).json()).access_token
  assert.ok(accessToken)
  await page.getByRole('region', { name: '综合态势', exact: true }).waitFor()
  await waitReady()
  await page.getByRole('combobox', { name: '数据源', exact: true }).selectOption(config.sourceId)
  await waitReady()
  await page.getByRole('combobox', { name: '已有会话', exact: true }).selectOption(config.session_id)
  await waitReady()
  await page.getByRole('combobox', { name: '监测点', exact: true }).selectOption(config.pointId)
  await waitReady()
  const baselineRuns = await api(`/api/forecasts?${new URLSearchParams({ session_id: config.session_id, point_id: config.pointId, limit: '20' })}`)
  const baseline = baselineRuns.find(run => run.model_type === 'baseline' && run.status === 'completed')
  assert.ok(baseline)
  report.baseline = await checkForecast(baseline)
  report.checks.push('Restored baseline matches persisted real-data forecast')

  const modelSelect = page.getByRole('combobox', { name: '预测模型', exact: true })
  await modelSelect.selectOption(`${config.modelId}:${config.version}`)
  await page.waitForFunction(() => !document.querySelector('.forecast-toolbar button')?.disabled)
  const forecastResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/forecasts' && response.request().method() === 'POST', { timeout: 90000 })
  await page.getByRole('button', { name: '运行预测', exact: true }).click()
  const response = await forecastResponse
  assert.equal(response.status(), 200)
  const submitted = response.request().postDataJSON()
  assert.equal(submitted.model_id, config.modelId)
  assert.equal(submitted.version, config.version)
  const professional = await response.json()
  await waitReady()
  await page.waitForFunction(() => !document.querySelector('.forecast-toolbar button')?.disabled)
  report.professional = await checkForecast(professional)
  assert.equal(professional.input_refs.length, 2700)
  report.checks.push('Selected professional version runs against 2700 real inputs; 12 rendered values match HTTP response')

  const eventButton = page.getByRole('button', { name: `实测超限 ${config.pointId} 第${config.event_revision}版 查看研判`, exact: true })
  const evidenceResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/evidence')
  await eventButton.click()
  const evidence = await (await evidenceResponse).json()
  await waitReady()
  assert.equal(evidence.event_snapshot.event_id, config.event_id)
  assert.equal(evidence.event_snapshot.event_revision, config.event_revision)
  assert.equal(evidence.candidate_attribution.status, 'completed')
  assert.equal(evidence.candidate_attribution.input_refs.length, 366)
  const attributionCard = page.getByRole('region', { name: '溯源结果', exact: true })
  await attributionCard.getByText('已完成', { exact: true }).waitFor()
  const attributionText = await attributionCard.innerText()
  for (const cause of evidence.candidate_attribution.candidate_causes) assert.ok(attributionText.includes(cause))
  assert.ok(attributionText.includes(evidence.candidate_attribution.model_version))
  assert.ok(attributionText.includes(mineTime(evidence.candidate_attribution.input_as_of)))
  assert.ok(attributionText.includes('366'))
  report.attribution = { run_id: evidence.candidate_attribution.run_id, model_version: evidence.candidate_attribution.model_version,
    event_id: config.event_id, event_revision: config.event_revision, input_reference_count: 366,
    target_time: evidence.candidate_attribution.target_time, input_as_of: evidence.candidate_attribution.input_as_of }
  const repeatedResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/evidence')
  await eventButton.click()
  const repeated = await (await repeatedResponse).json()
  assert.equal(repeated.candidate_attribution.run_id, report.attribution.run_id)
  assert.equal(repeated.candidate_attribution.reused, true)
  await waitReady()
  report.checks.push('Real observed event binds matching revision and six-channel attribution; repeated selection reuses run')

  const activeSession = await api(`/api/sessions/${encodeURIComponent(config.session_id)}`)
  const observations = await api(`/api/observations?${new URLSearchParams({ session_id: config.session_id, point_id: config.pointId,
    start_at: new Date(Date.parse(activeSession.as_of) - 3600000).toISOString(), end_at: activeSession.as_of, aggregation_seconds: '60' })}`)
  const lastReading = observations.rows.filter(row => Number.isFinite(row.value)).at(-1)
  assert.ok(lastReading)
  report.displayed_times = { selected_reading_time: lastReading.event_time, data_as_of: activeSession.as_of }

  for (const viewport of [{ width: 1920, height: 1080 }, { width: 1440, height: 900 }, { width: 1280, height: 800 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport)
    await page.evaluate(() => document.fonts.ready)
    const dimensions = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }))
    assert.ok(dimensions.page <= dimensions.viewport + 1, `Horizontal overflow at ${viewport.width}px`)
    const channelLayout = await page.locator('.channel-grid').evaluate(grid => {
      const bounds = grid.getBoundingClientRect()
      const cards = [...grid.querySelectorAll('.channel-card')]
      const fits = rect => rect.top >= bounds.top - 1 && rect.bottom <= bounds.bottom + 1
        && rect.left >= bounds.left - 1 && rect.right <= bounds.right + 1
      const times = cards.flatMap(card => [...card.querySelectorAll('.channel-time')])
      const visibleTimes = times.filter(element => getComputedStyle(element).display !== 'none' && element.getClientRects().length)
      return { cards: cards.length, client_height: grid.clientHeight, scroll_height: grid.scrollHeight,
        scroll_height_difference: Math.max(0, grid.scrollHeight - grid.clientHeight),
        all_cards_visible: cards.every(card => fits(card.getBoundingClientRect())),
        all_values_and_units_visible: cards.every(card => [...card.querySelectorAll('strong,strong small')].every(element => fits(element.getBoundingClientRect()))),
        visible_time_count: visibleTimes.length, hidden_time_count: times.length - visibleTimes.length,
        all_visible_times_fit: visibleTimes.every(element => fits(element.getBoundingClientRect())) }
    })
    assert.equal(channelLayout.cards, 6)
    assert.ok(channelLayout.all_cards_visible && channelLayout.all_values_and_units_visible && channelLayout.all_visible_times_fit, `Channel fields clipped at ${viewport.width}px`)
    assert.equal(channelLayout.hidden_time_count, viewport.width <= 430 ? 6 : 0)
    assert.ok((await page.locator('.summary-card').first().innerText()).includes(mineTime(lastReading.event_time)), 'Selected reading time remains visible')
    assert.ok((await page.locator('.channels-panel footer').innerText()).includes(mineTime(activeSession.as_of)), 'Shared data clock remains visible')
    await page.screenshot({ path: resolve(output, `real-data-${viewport.width}.png`), fullPage: true, mask: screenshotMask(), maskColor: '#0b1e30' })
    report.viewports.push({ ...viewport, ...dimensions, horizontal_overflow: false, channels: channelLayout })
  }
  report.checks.push('1920/1440/1280/390 pixel viewports have no document horizontal overflow')
  report.checks.push('All six cards, values, units and visible time fields fit; mobile retains selected reading time and shared data clock')
  await page.setViewportSize({ width: 1440, height: 900 })

  const beforeWind = requests.length
  await page.getByRole('combobox', { name: '监测点', exact: true }).selectOption(config.windPointId)
  await waitReady()
  await page.getByRole('heading', { name: '工况趋势', exact: true }).waitFor()
  assert.equal(await modelSelect.isDisabled(), true)
  assert.equal(await page.getByRole('region', { name: '溯源结果', exact: true }).count(), 0)
  assert.equal(await page.getByRole('table', { name: '预测区间数据', exact: true }).count(), 0)
  assert.equal(requests.slice(beforeWind).some(request => ['/api/models', '/api/rules', '/api/events', '/api/forecasts'].includes(request.path)), false)
  report.checks.push('Wind monitoring clears methane predictions, thresholds and attribution without methane API calls')

  await page.getByRole('combobox', { name: '数据源', exact: true }).selectOption(config.otherSourceId)
  await waitReady()
  assert.equal(await page.getByRole('combobox', { name: '已有会话', exact: true }).inputValue(), '')
  assert.equal(await page.getByRole('region', { name: '溯源结果', exact: true }).count(), 0)
  assert.equal(await page.getByRole('table', { name: '预测区间数据', exact: true }).count(), 0)
  await page.getByRole('combobox', { name: '已有会话', exact: true }).selectOption(config.session_id)
  await waitReady()
  assert.equal(await page.getByRole('combobox', { name: '数据源', exact: true }).inputValue(), config.sourceId)
  await page.getByRole('combobox', { name: '监测点', exact: true }).selectOption(config.pointId)
  await waitReady()
  await modelSelect.selectOption(`${config.modelId}:${config.version}`)
  await page.waitForFunction(() => !document.querySelector('.forecast-toolbar button')?.disabled)
  const restored = await api(`/api/forecasts?${new URLSearchParams({ session_id: config.session_id, point_id: config.pointId, model_id: config.modelId, version: config.version, limit: '1' })}`)
  assert.equal(restored[0].run_id, professional.run_id)
  await checkForecast(restored[0])
  report.checks.push('Changing source clears stale results; restoring session and model reloads persisted professional output')
  assert.deepEqual(pageErrors, [])
  assert.deepEqual(foreignRequests, [])
  report.status = 'passed'
  report.page_errors = pageErrors
  report.external_requests = foreignRequests.length
} catch (error) {
  report.status = 'failed'
  report.error = safeMessage(error.message)
  await page.screenshot({ path: resolve(output, 'real-data-failure.png'), fullPage: true, mask: screenshotMask(), maskColor: '#0b1e30' }).catch(() => {})
  process.exitCode = 1
} finally {
  await writeFile(resolve(output, 'browser-report.json'), JSON.stringify(report, null, 2))
  await context.close()
  await browser.close()
}
console.log(JSON.stringify({ status: report.status, checks: report.checks.length, report: resolve(output, 'browser-report.json') }))
