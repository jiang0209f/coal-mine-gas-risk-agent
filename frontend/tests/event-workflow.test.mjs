import assert from 'node:assert/strict'
import test from 'node:test'

const helpers = () => import('../src/event-workflow.mjs')

test('workflow requests bind the current session and encode event and report identifiers', async () => {
  const { createEventWorkflowClient } = await helpers()
  const calls = []
  const client = createEventWorkflowClient(async (...args) => { calls.push(args); return {} }, 'session-1', 'event/a')
  await client.loadInvestigation()
  await client.startInvestigation()
  await client.reviewInvestigation('report/a', 'approve', '依据已核对')
  await client.downloadReport('report/a')
  assert.deepEqual(calls, [
    ['/events/event%2Fa/investigation?session_id=session-1'],
    ['/events/event%2Fa/investigation', { method: 'POST', body: { session_id: 'session-1' } }],
    ['/investigations/report%2Fa/review', { method: 'POST', body: { session_id: 'session-1', decision: 'approve', note: '依据已核对' } }],
    ['/investigations/report%2Fa/report.docx?session_id=session-1', { responseType: 'blob' }],
  ])
})

test('control actions use explicit human submissions and retain the report link', async () => {
  const { createEventWorkflowClient } = await helpers()
  const calls = []
  const client = createEventWorkflowClient(async (...args) => { calls.push(args); return {} }, 's', 'e')
  await client.importMeasures(['ref-1', 'ref-2'])
  await client.createPlan('report-1')
  await client.executeTask('task-1', ' 已核对风量 ')
  await client.verifyPlan('plan-1')
  await client.reviewCompletion('plan-1', '核验结果已复核')
  await client.reviewExperience('experience-1', 'reject', '补充适用条件')
  assert.deepEqual(calls, [
    ['/measures', { method: 'POST', body: { session_id: 's', reference_ids: ['ref-1', 'ref-2'] } }],
    ['/events/e/control-plans', { method: 'POST', body: { session_id: 's', report_id: 'report-1' } }],
    ['/control-tasks/task-1/execute', { method: 'POST', body: { session_id: 's', content: '已核对风量' } }],
    ['/control-plans/plan-1/verify', { method: 'POST', body: { session_id: 's' } }],
    ['/control-plans/plan-1/review-completion', { method: 'POST', body: { session_id: 's', note: '核验结果已复核' } }],
    ['/control-experiences/experience-1/review', { method: 'POST', body: { session_id: 's', decision: 'reject', note: '补充适用条件' } }],
  ])
})

test('invalid review decisions empty execution and missing context never issue requests', async () => {
  const { createEventWorkflowClient } = await helpers()
  let requests = 0
  const api = async () => { requests++; return {} }
  assert.throws(() => createEventWorkflowClient(api, '', 'e'), /会话/)
  const client = createEventWorkflowClient(api, 's', 'e')
  await assert.rejects(() => client.reviewPlan('p', 'approved', '核对'), /审核/)
  await assert.rejects(() => client.executeTask('t', ' '), /执行记录/)
  await assert.rejects(() => client.createPlan(''), /排查报告/)
  assert.equal(requests, 0)
})

test('report wrappers and asynchronous states remain distinct from review approval', async () => {
  const { unwrapInvestigation, workflowState, shouldPollInvestigation, canWorkflowReview } = await helpers()
  const report = { report_id: 'r', state: 'draft' }
  assert.equal(unwrapInvestigation({ report, history: [] }), report)
  assert.equal(unwrapInvestigation(report), report)
  assert.equal(unwrapInvestigation({ report: null, history: [] }), null)
  assert.equal(workflowState(report), 'draft')
  assert.equal(shouldPollInvestigation({ state: 'running' }), true)
  assert.equal(shouldPollInvestigation(report), false)
  assert.equal(canWorkflowReview(report, ['safety_management']), true)
  assert.equal(canWorkflowReview(report, ['ventilation']), false)
  assert.equal(canWorkflowReview({ state: 'approved' }, ['safety_management']), false)
})

test('task execution matches the responsible role and overdue uses the session clock', async () => {
  const { canExecuteControlTask, isControlTaskOverdue } = await helpers()
  const task = { state: 'pending', responsible_role: 'ventilation', due_at: '2026-11-05T01:00:00Z' }
  assert.equal(canExecuteControlTask(task, ['ventilation'], '已核实', 'approved'), true)
  assert.equal(canExecuteControlTask(task, ['gas_governance'], '已核实', 'approved'), false)
  assert.equal(canExecuteControlTask(task, ['safety_management'], '已核实', 'approved'), true)
  assert.equal(canExecuteControlTask(task, ['ventilation'], '已核实', 'draft'), false)
  assert.equal(canExecuteControlTask(task, ['ventilation'], '', 'approved'), false)
  assert.equal(isControlTaskOverdue(task, '2026-11-05T01:00:01Z'), true)
  assert.equal(isControlTaskOverdue(task, '2026-11-05T01:00:00Z'), false)
  assert.equal(isControlTaskOverdue({ ...task, state: 'executed' }, '2026-11-05T02:00:00Z'), false)
  assert.equal(isControlTaskOverdue({ ...task, overdue: true }, '2026-11-05T00:00:00Z'), true)
})

test('hypothesis assessment retains opposing and inconclusive evidence', async () => {
  const { assessmentLabel, parseReferenceIds } = await helpers()
  assert.equal(assessmentLabel({ support_status: 'supported' }), '支持')
  assert.equal(assessmentLabel({ assessment: 'oppose' }), '反对')
  assert.equal(assessmentLabel({ support_status: 'undetermined' }), '无法判断')
  assert.deepEqual(parseReferenceIds(' ref-1，ref-2\nref-1 ; ref-3 '), ['ref-1', 'ref-2', 'ref-3'])
})

test('review deadlines use mine UTC+8 and measure tiers retain the explicit human choice', async () => {
  const { createEventWorkflowClient, planDeadlines } = await helpers()
  const due = planDeadlines([{ measure_id: 'm1' }, { measure_id: 'm2' }], { m1: '2026-11-05T15:30:00', m2: '' })
  assert.deepEqual(due, { m1: '2026-11-05T07:30:00.000Z' })
  assert.throws(() => planDeadlines([{ measure_id: 'm1' }], { m1: 'invalid' }), /矿区时间/)
  const calls = []
  const client = createEventWorkflowClient(async (...args) => { calls.push(args); return {} }, 's', 'e')
  await client.reviewMeasure('m1', 'approve', '核对原文及措施层级', 3)
  await client.reviewPlan('p1', 'approve', '时限已核对', due)
  assert.equal(calls[0][1].body.tier, 3)
  assert.deepEqual(calls[1][1].body.task_due_at, due)
  await assert.rejects(() => client.reviewMeasure('m1', 'approve', '核对', 6), /层级/)
  assert.equal(calls.length, 2)
})

test('measure review submits the responsible role explicitly chosen by the reviewer', async () => {
  const { createEventWorkflowClient } = await helpers()
  const calls = []
  const client = createEventWorkflowClient(async (...args) => { calls.push(args); return {} }, 's', 'e')
  await client.reviewMeasure('m1', 'approve', '责任角色已确认', 2, 'ventilation')
  assert.deepEqual(calls[0], ['/measures/m1/review', { method: 'POST', body: {
    session_id: 's', decision: 'approve', note: '责任角色已确认', tier: 2, assigned_role: 'ventilation',
  } }])
  await assert.rejects(() => client.reviewMeasure('m1', 'approve', '核对', 2, '待矿方确定'), /责任角色/)
  assert.equal(calls.length, 1)
})
