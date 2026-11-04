const statuses = { disabled: '未启用', awaiting_data: '等待数据', receiving: '近期有数据', active: '近期有数据', stale: '数据中断', retry_wait: '等待重试', error: '待核查' }
export function connectorStatusLabel(connector) {
  if (!connector?.enabled) return '未启用'
  return statuses[connector.status || connector.state] || '等待数据'
}
export function workerStatusLabel(worker) {
  if (!worker) return '读取中'
  return { running: '运行中', starting: '启动中', disabled: '已停用', degraded: '异常', stopped: '已停止' }[worker.state] || '待核查'
}
export function batchStatusLabel(batch) {
  const status = typeof batch === 'string' ? batch : batch?.status
  return { pending: '待入库', queued: '待入库', processing: '入库中', completed: '已入库', retry_wait: '等待重试', retrying: '等待重试', rejected: '待核查', conflict: '内容冲突' }[status] || '待核查'
}
export function ingestionErrorLabel(code) {
  if (!code) return ''
  return {
    reading_conflict: '测值编号对应的内容冲突', reading_id_conflict: '测值编号对应的内容冲突', batch_id_conflict: '批次编号对应的内容冲突',
    authentication_required: '接入身份需要重新认证', storage_unavailable: '存储暂不可用，等待重试', ingest_unavailable: '入库暂不可用，等待重试', invalid_readings: '监测记录需要核对',
    connector_disabled: '接入已暂停', invalid_payload: '批次格式需要核对', invalid_reading: '监测记录需要核对',
    unknown_point: '测点尚未登记', point_source_mismatch: '测点与数据来源不一致', unmapped_tag: '上游标签尚未映射',
    future_event_time: '监测时间晚于接收时间', invalid_event_time: '监测时间格式需要核对',
    invalid_value: '监测值格式需要核对', invalid_mapping: '字段映射需要核对', payload_too_large: '批次超过接收上限',
  }[code] || '接入处理未完成'
}
const clean = value => String(value ?? '').trim()
export function ingestionSpec(form, sourceId, points = []) {
  if (!sourceId) throw new Error('请先选择数据来源。')
  const name = clean(form.name)
  if (!name) throw new Error('请填写接入名称。')
  const adapter = form.adapter || 'standard_json'
  if (!['standard_json', 'mapped_json'].includes(adapter)) throw new Error('请选择支持的数据格式。')
  const spec = { name, source_id: sourceId, adapter_kind: adapter, enabled: false, mapping: {}, tag_map: {} }
  if (clean(form.connectorId)) spec.connector_id = clean(form.connectorId)
  const seconds = Number(form.staleSeconds ?? 120)
  if (!Number.isInteger(seconds) || seconds < 1) throw new Error('数据中断判定时长应为正整数秒。')
  spec.stale_after_seconds = seconds
  if (adapter === 'mapped_json') {
    for (const [key, field] of [['tag_field', 'tagField'], ['time_field', 'timeField'], ['value_field', 'valueField']]) {
      if (!clean(form[field])) throw new Error('请补齐标签、时间和数值字段。')
      spec.mapping[key] = clean(form[field])
    }
    for (const [key, field] of [['records_path', 'rowsPath'], ['id_field', 'idField'], ['sequence_field', 'sequenceField'], ['timezone', 'timezone']]) {
      if (clean(form[field])) spec.mapping[key] = clean(form[field])
    }
    if (!form.bindings?.length) throw new Error('请添加上游标签与测点映射。')
    for (const binding of form.bindings) {
      const tag = clean(binding.tag)
      if (!tag) throw new Error('请填写每条映射的上游标签。')
      if (Object.hasOwn(spec.tag_map, tag)) throw new Error('上游标签重复，请合并映射。')
      if (!points.some(point => point.point_id === binding.pointId && point.source_id === sourceId)) throw new Error('请将上游标签映射到当前来源的已登记测点。')
      Object.defineProperty(spec.tag_map, tag, { value: binding.pointId, enumerable: true, configurable: true, writable: true })
    }
  }
  return spec
}
