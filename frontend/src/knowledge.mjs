export function referenceLocation(locator = {}) {
  const parts = []
  if (locator.page) parts.push(`第${locator.page}页`)
  if (locator.paragraph) parts.push(`第${locator.paragraph}段`)
  if (locator.table) parts.push(`表${locator.table}`)
  if (locator.row) parts.push(`第${locator.row}行`)
  if (locator.line) parts.push(`第${locator.line}行`)
  const chunk = locator.chunk ?? locator.part, start = locator.start_offset ?? locator.char_start, end = locator.end_offset ?? locator.char_end
  if (chunk) parts.push(`片段${chunk}`)
  if (Number.isInteger(start) && Number.isInteger(end) && end > start) parts.push(`字符${start + 1}—${end}`)
  if (locator.section) parts.push(locator.section)
  if (locator.case_id) parts.push(`处置单 ${locator.case_id}`)
  return parts.join(' · ') || '原文片段'
}

export function referenceValidity(entry, now = Date.now()) {
  const start = Date.parse(entry?.applicability?.valid_from), end = Date.parse(entry?.applicability?.valid_to)
  if (Number.isFinite(start) && start > now) return 'not_effective'
  if (Number.isFinite(end) && end <= now) return 'expired'
  return 'current'
}

export function adviceIsStale(packet, advice) {
  if (!advice) return false
  const validity = packet?.advice_validity?.[`${advice.advice_id}:${advice.version}`]
  if (validity) return validity.stale === true
  return packet?.current_event_revision !== undefined && advice.event_revision !== undefined
    && packet.current_event_revision !== advice.event_revision
}

export function canCaseAction(action, { packet, advice, roles = [], content = '', busy = false } = {}) {
  if (busy || !roles.some(role => ['ventilation', 'gas_governance', 'safety_management'].includes(role))) return false
  const phase = packet?.case?.phase || 'new'
  if (action === 'confirm_event') return phase === 'new' && Boolean(content.trim())
  if (action === 'submit_feedback') return phase === 'executed' && Boolean(content.trim())
  if (action === 'review_case') return phase === 'feedback_submitted' && Boolean(content.trim())
  if (!advice || adviceIsStale(packet, advice)) return false
  const decision = packet?.advice_review_status?.[`${advice.advice_id}:${advice.version}`]?.decision
  if (action === 'review_advice') return roles.includes('safety_management') && ['confirmed', 'advice_reviewed'].includes(phase) && !decision
  if (action === 'record_execution') return phase === 'advice_reviewed' && decision === 'approved' && Boolean(content.trim())
  return false
}
