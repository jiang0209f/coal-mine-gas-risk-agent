// 隔离接口夹具验证界面操作；测试数据不进入产品，也不调用真实模型。
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { once } from 'node:events'
import { readFile, mkdir } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { resolve, extname } from 'node:path'
import test from 'node:test'
import { chromium } from 'playwright'
import { mineTime } from '../src/utils.mjs'

const root = resolve(import.meta.dirname, '..', 'dist')
const artifactDir = resolve(import.meta.dirname, 'artifacts')
const browserPath = process.env.UI_TEST_BROWSER || [
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
].find(path => existsSync(path))
const source = { source_id: 'qa-source', name: '界面验收数据（测试夹具）', face_id: 'TEST-UI' }
const point = { point_id: 'qa-T0', source_id: source.source_id, face_id: source.face_id, name: 'T0（界面验收）', unit: '%CH4', unit_status: 'inferred' }
const reference = { reference_id: 'qa-approved-reference', kind: 'document', source: 'UI验收资料', locator: { page: 1 }, text: '测试条文：核对监测记录。', operation_context: 'field_record' }
let role, session, actions, advice, calls, pendingChat, taskState, attributionState
function reset(value) { role = value; session = null; actions = []; advice = []; calls = []; pendingChat = false; taskState = 'running'; attributionState = 'completed' }
function event() {
  return { event_id: 'qa-event', session_id: session.session_id, point_id: point.point_id, event_revision: 1,
    start_at: new Date(Date.parse(session.as_of) - 60000).toISOString(), event_type: 'observed_exceedance',
    alert_level: 'red', quality_flags: ['suspected_calibration'] }
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
    advice_review_status: review ? { 'qa-advice:1': { decision: review.review_decision } } : {}, pending_items: [] }
}
async function api(path, request) {
  const chunks = []
  for await (const chunk of request) chunks.push(chunk)
  const text = Buffer.concat(chunks).toString()
  const body = text ? JSON.parse(text) : null
  calls.push({ path: path.pathname, query: Object.fromEntries(path.searchParams), method: request.method, body })
  if (path.pathname === '/api/auth/login') return { access_token: 'qa-isolated-token', token_type: 'bearer', user: { user_id: 'qa-user', username: '界面验收账户', roles: [role], face_ids: [source.face_id] } }
  if (path.pathname === '/api/auth/logout') return {}
  if (path.pathname === '/api/sources') return [source]
  if (path.pathname === '/api/points') return [point]
  if (path.pathname === '/api/references') return []
  if (path.pathname === '/api/evaluations') return []
  if (path.pathname === '/api/sessions' && request.method === 'POST') {
    session = { ...body, session_id: 'qa-session', face_id: source.face_id, as_of: body.start_as_of, state: 'paused', speed: 1 }
    return session
  }
  if (path.pathname === '/api/sessions') return session ? [session] : []
  if (path.pathname === '/api/sessions/qa-session') return session
  if (path.pathname === '/api/sessions/qa-session/actions') {
    session = { ...session, state: body.action === 'pause' ? 'paused' : 'running' }
    return session
  }
  if (path.pathname === '/api/sessions/qa-session/tick') {
    session = { ...session, as_of: body.target_as_of || new Date(Date.parse(session.as_of) + 60000).toISOString() }
    return { session, status: 'completed' }
  }
  if (path.pathname === '/api/observations') return { point_id: point.point_id, as_of: session.as_of, unit: '%CH4', total: 3,
    quality: { valid_rows: 1, invalid_rows: 1 }, rows: [
      { event_time: new Date(Date.parse(session.as_of) - 120000).toISOString(), value: 0.91 },
      { event_time: new Date(Date.parse(session.as_of) - 60000).toISOString(), value: null },
      { event_time: new Date(Date.parse(session.as_of) + 60000).toISOString(), value: 9.99 },
    ] }
  if (path.pathname === '/api/events') return [event()]
  if (path.pathname === '/api/rules') return { rule_version: 'qa-draft', status: 'team-draft', source: 'UI验收规则（测试夹具）', observed_red_threshold: 0.8, observed_orange_threshold: 0.65, orange_duration_seconds: 300, forecast_yellow_threshold: 0.65 }
  if (path.pathname === '/api/models') return [{ model_id: 'QA离线基线', version: 'test', model_type: 'baseline', input_window: 2, horizon_seconds: 60 }]
  if (path.pathname === '/api/evidence') return { event_snapshot: event(), snapshot_as_of: session.as_of,
    observation_fragments: [], condition_hints: [{ kind: 'suspected_calibration', text: 'UI验收：疑似标校提示。' }], missing_fields: [],
    candidate_attribution: candidateAttribution() }
  if (path.pathname === '/api/cases/qa-event') return casePacket()
  if (path.pathname === '/api/cases/qa-event/actions') {
    actions.push({ ...body, action_id: `qa-action-${actions.length}`, actor_id: 'qa-user', occurred_at: new Date().toISOString(), data_as_of: session.as_of })
    return { case: casePacket().case, case_action: actions.at(-1) }
  }
  if (path.pathname === '/api/references/search') return [reference]
  if (path.pathname === '/api/advice') {
    advice.push({ advice_id: 'qa-advice', version: 1, reference_ids: body.reference_ids, items: [reference] })
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
  try {
    for (const userRole of ['ventilation', 'gas_governance', 'safety_management']) {
      await t.test(`${userRole}: login, scoped data, chart and operation permissions`, async () => {
        reset(userRole)
        const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } })
        const page = await context.newPage()
        page.setDefaultTimeout(6000)
        const failures = []; page.on('pageerror', error => failures.push(error.message))
        await page.clock.install()
        await page.goto(url)
        assert.equal(calls.length, 0, '登录前不请求业务资料')
        await page.locator('[name="username"]').fill('qa-only-user')
        await page.locator('[name="password"]').fill('qa-only-password-for-test')
        await page.getByRole('button', { name: /^登录工作台/ }).click()
        await page.getByText('TEST-UI 工作面', { exact: true }).waitFor()
        await page.getByRole('button', { name: '创建会话', exact: true }).click()
        await page.locator('.metric-card strong').first().filter({ hasText: '0.910' }).waitFor()
        assert.ok(calls.filter(call => call.path === '/api/observations').every(call => call.query.end_at === session.as_of))
        assert.equal(await page.getByRole('button', { name: '批准建议', exact: true }).count(), 0)
        assert.equal(calls.some(call => call.path === '/api/evaluations'), userRole === 'safety_management')
        await page.getByRole('button', { name: /实测超限/ }).click()
        await page.getByPlaceholder('填写人工核实、执行、反馈或复核记录').waitFor()
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
          await attributionCard.getByText('开启事件溯源', { exact: true }).waitFor()
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
          const chartBounds = await page.locator('.chart').evaluate(element => ({ parent: element.clientWidth, canvas: element.querySelector('canvas')?.clientWidth }))
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
        await context.close()
      })
    }
  } finally { await browser.close(); await new Promise(resolveClose => server.close(resolveClose)) }
})
