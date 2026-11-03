function awareDate(value) {
  if (typeof value !== 'string' || !/(Z|[+-]\d{2}:\d{2})$/.test(value)) throw new Error('时间需要包含时区')
  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) throw new Error('时间格式无效')
  return date
}

export function advanceAsOf(value, seconds = 60) {
  return new Date(awareDate(value).getTime() + seconds * 1000).toISOString()
}

export function chinaInputToIso(value) {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(value)) throw new Error('请选择有效的矿区时间')
  return awareDate(`${value}+08:00`).toISOString()
}

export function observationWindow(asOf, minutes = 60) {
  const end = awareDate(asOf)
  return { start_at: new Date(end.getTime() - minutes * 60000).toISOString(), end_at: end.toISOString() }
}

export function seriesToPoints(packet, asOf) {
  const cutoff = awareDate(asOf).getTime()
  const rows = Array.isArray(packet) ? packet : packet?.rows || []
  return rows.filter(row => Number.isFinite(Date.parse(row.event_time)) && Date.parse(row.event_time) <= cutoff)
    .map(row => [Date.parse(row.event_time), typeof row.value === 'number' && Number.isFinite(row.value) ? row.value : null])
    .sort((a, b) => a[0] - b[0])
}

export function mineTime(value, seconds = true) {
  if (value === undefined || value === null || value === '' || !Number.isFinite(new Date(value).getTime())) return '—'
  return new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', ...(seconds ? { second: '2-digit' } : {}), hour12: false,
  }).format(new Date(value)).replaceAll('/', '-')
}

export function queryString(fields) {
  return new URLSearchParams(Object.entries(fields).filter(([, value]) => value !== undefined && value !== null && value !== '')).toString()
}
