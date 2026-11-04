import { chinaInputToIso } from './utils.mjs'

export const metricNames = { precision: '逐事件查准率', recall: '逐事件召回率', missed_events: '漏报事件数', false_alerts_per_valid_monitoring_hour: '每有效监测点小时误报数', sample_precision: '逐样本查准率', sample_recall: '逐样本召回率', sample_fpr: '逐样本误报率', mae: '平均绝对误差', mape_with_zero_policy: '非零样本平均百分比误差', tool_task_completion_rate: '工具任务完成率', evidence_completeness: '依据完整率', advice_reference_coverage: '建议出处覆盖率' }
const reasons = { no_observed_assessments: '未保存会话实测阈值事件评估记录', zero_denominator: '分母为零', no_actual_events: '没有实测阈值事件', no_positive_events: '没有实测阈值事件', no_alert_events: '没有预测预警事件', no_positive_samples: '没有实测正样本', no_predicted_positive_samples: '没有预测正样本', no_negative_samples: '没有实测负样本', no_regression_samples: '没有有效的实测与预测配对', zero_actuals_excluded: '实测值均为零，未计算百分比误差', no_tasks: '没有符合范围的工具任务', no_advice_items: '没有建议条目', missing_valid_monitoring_hours: '缺少有效监测时长', no_valid_monitoring_hours: '没有有效监测时长', no_mature_forecasts: '没有已成熟的预测区间', no_mature_forecast_samples: '没有已成熟的预测区间', no_forecast_runs: '没有已保存的预测运行', forecast_event_policy_not_supported: '当前预测预警口径未纳入事件评估' }

export function metricDisplay(value) {
  return value === null || value === undefined ? '未计算' : typeof value === 'number' ? String(Number(value.toFixed(6))) : String(value)
}

export function evaluationReason(status = {}) {
  const reason = status.reason || status.undefined_reason
  return reasons[reason] || (typeof reason === 'string' && /[\u4e00-\u9fff]/.test(reason) ? reason : reason ? '当前记录不足，未计算' : '')
}

export function replaySessionSpec(session, target) {
  if (!session || !['historical_replay', 'simulated_stream'].includes(session.mode)) throw new Error('重新回放需要历史或模拟会话')
  const instant = Date.parse(target)
  if (!Number.isFinite(instant)) throw new Error('请选择有效的回放起点')
  if (instant > Date.parse(session.as_of)) throw new Error('回放起点不能晚于当前数据时间')
  return { source_id: session.source_id, mode: session.mode, start_as_of: new Date(instant).toISOString(),
    config_version: session.config_version, speed: session.speed || 1, parent_session_id: session.session_id }
}

export function evaluationRequest({ session, startTime, endTime, pointIds, model, predictionStage, datasetSplit = 'unclassified', toleranceSeconds = 0 }) {
  if (!session?.session_id) throw new Error('请先选择会话')
  const start = chinaInputToIso(startTime), end = chinaInputToIso(endTime)
  if (Date.parse(end) <= Date.parse(start)) throw new Error('结束时间需要晚于开始时间')
  if (Date.parse(end) > Date.parse(session.as_of)) throw new Error('结束时间不能晚于当前数据时间')
  return { session_id: session.session_id, start_at: start, end_at: end,
    ...(pointIds?.length ? { point_ids: pointIds } : {}), ...(model ? { model_id: model.model_id, model_version: model.version || model.model_version } : {}),
    ...(predictionStage ? { prediction_stage: predictionStage } : {}), dataset_split: datasetSplit, tolerance_seconds: Number(toleranceSeconds) }
}

export function detailRowLabel(row = {}) {
  const record = row.record || row.actual_event || row.alert_event || row
  const name = { TP: '命中样本', FP: '误报样本', TN: '正确负样本', FN: '漏报样本' }[row.classification]
    || { event_match: '事件匹配', actual_event: row.matched ? '已匹配实测事件' : '未匹配实测事件', alert_event: row.matched ? '已匹配预测事件' : '未匹配预测事件', unmatched_actual: '未匹配实测事件', unmatched_alert: '未匹配预测事件', sample: '预测配对样本', task: '工具任务', advice: '建议出处', monitoring_interval: '有效监测区间' }[row.kind] || '冻结记录'
  return [name, record.sample_id || record.event_id || record.task_id || record.advice_id || record.point_id].filter(Boolean).join(' · ')
}
