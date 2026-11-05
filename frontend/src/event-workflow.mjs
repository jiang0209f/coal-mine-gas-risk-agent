import { chinaInputToIso, queryString } from './utils.mjs'

const businessRoles = ['ventilation', 'gas_governance', 'safety_management']

export function workflowState(item) {
  return item?.state || item?.status || item?.review_state || 'not_started'
}

export function unwrapInvestigation(packet) {
  if (!packet || typeof packet !== 'object') return null
  return Object.hasOwn(packet, 'report') ? packet.report : packet
}

export function shouldPollInvestigation(report) {
  return ['queued', 'running'].includes(workflowState(report))
}

export function canWorkflowReview(item, roles = [], busy = false) {
  return !busy && roles.includes('safety_management') && ['draft', 'pending', 'pending_review'].includes(workflowState(item))
}

export function canExecuteControlTask(task, roles = [], content = '', planState = '', busy = false) {
  const responsible = task?.responsible_role || task?.assigned_role || task?.role
  return !busy && ['approved', 'executing', 'in_progress'].includes(planState)
    && ['pending', 'assigned', 'new', 'ready'].includes(workflowState(task))
    && roles.some(role => businessRoles.includes(role) && (role === 'safety_management' || !responsible || responsible === role))
    && Boolean(content.trim())
}

export function isControlTaskOverdue(task, asOf) {
  if (['executed', 'completed', 'verified', 'cancelled'].includes(workflowState(task))) return false
  if (typeof task?.overdue === 'boolean') return task.overdue
  const deadline = Date.parse(task?.due_at || task?.deadline), now = Date.parse(asOf)
  return Number.isFinite(deadline) && Number.isFinite(now) && now > deadline
}

export function assessmentLabel(item) {
  return { supported: '支持', support: '支持', opposed: '反对', oppose: '反对', undetermined: '无法判断', unknown: '无法判断' }[item?.support_status || item?.assessment] || '无法判断'
}

export function parseReferenceIds(value = '') {
  return [...new Set(value.split(/[\s,，;；]+/).map(item => item.trim()).filter(Boolean))]
}

export function planDeadlines(items = [], input = {}) {
  return Object.fromEntries(items.filter(item => input[item.measure_id]?.trim())
    .map(item => [item.measure_id, chinaInputToIso(input[item.measure_id].trim())]))
}

export function createEventWorkflowClient(api, sessionId, eventId) {
  if (typeof api !== 'function' || !sessionId) throw new Error('请先选择有效会话')
  if (!eventId) throw new Error('请先选择事件')
  const id = value => encodeURIComponent(value)
  const scoped = path => `${path}?${queryString({ session_id: sessionId })}`
  const get = path => api(scoped(path))
  const post = (path, body = {}) => api(path, { method: 'POST', body: { session_id: sessionId, ...body } })
  const review = async (path, decision, note, extra = {}) => {
    if (!['approve', 'reject'].includes(decision)) throw new Error('请选择有效的审核决定')
    if (!note?.trim()) throw new Error('请填写审核意见')
    return post(path, { decision, note: note.trim(), ...extra })
  }
  return {
    loadInvestigation: () => get(`/events/${id(eventId)}/investigation`),
    startInvestigation: (retry = false) => post(`/events/${id(eventId)}/investigation`, retry ? { retry: true } : {}),
    getInvestigation: reportId => get(`/investigations/${id(reportId)}`),
    reviewInvestigation: (reportId, decision, note) => review(`/investigations/${id(reportId)}/review`, decision, note),
    downloadReport: reportId => api(scoped(`/investigations/${id(reportId)}/report.docx`), { responseType: 'blob' }),
    loadControl: () => get(`/events/${id(eventId)}/control`),
    loadMeasures: () => get('/measures'),
    importMeasures: async referenceIds => {
      if (!Array.isArray(referenceIds) || !referenceIds.length) throw new Error('请选择措施引用')
      return post('/measures', { reference_ids: referenceIds })
    },
    reviewMeasure: async (measureId, decision, note, tier, assignedRole) => {
      if (tier !== undefined && (!Number.isInteger(tier) || tier < 1 || tier > 5)) throw new Error('措施层级须为1至5')
      if (assignedRole !== undefined && !businessRoles.includes(assignedRole)) throw new Error('请选择有效的责任角色')
      return review(`/measures/${id(measureId)}/review`, decision, note, {
        ...(tier === undefined ? {} : { tier }), ...(assignedRole === undefined ? {} : { assigned_role: assignedRole }),
      })
    },
    createPlan: async reportId => {
      if (!reportId) throw new Error('请选择已审核排查报告')
      return post(`/events/${id(eventId)}/control-plans`, { report_id: reportId })
    },
    getPlan: planId => get(`/control-plans/${id(planId)}`),
    reviewPlan: (planId, decision, note, taskDueAt) => review(`/control-plans/${id(planId)}/review`, decision, note, taskDueAt === undefined ? {} : { task_due_at: taskDueAt }),
    executeTask: async (taskId, content) => {
      if (!content?.trim()) throw new Error('请填写执行记录')
      return post(`/control-tasks/${id(taskId)}/execute`, { content: content.trim() })
    },
    verifyPlan: planId => post(`/control-plans/${id(planId)}/verify`),
    reviewCompletion: async (planId, note) => {
      if (!note?.trim()) throw new Error('请填写复核意见')
      return post(`/control-plans/${id(planId)}/review-completion`, { note: note.trim() })
    },
    reviewExperience: (experienceId, decision, note) => review(`/control-experiences/${id(experienceId)}/review`, decision, note),
  }
}
