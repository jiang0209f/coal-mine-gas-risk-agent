# 补传预警、连续监测与问答接入

本轮沿用现有 DeepSeek API、FastAPI、Vue 和本机模型推理环境，完成补传预警、预测预警联动、聊天中断恢复及服务端连续监测。环境、原始数据、权重、知识原件和运行报告放 F 盘。

## 补传后的事件与依据

预警服务按观测的 `available_at` 查找新可用的迟到记录，仅重建受影响时间段。日常扫描继续使用增量范围；数据库新增 `(point_id, available_at, event_time)` 索引，已有库启动时补建。

补传高值可新增事件或复用既有事件编号追加修订。补传正常值可关闭或拆分持续片段；既有事件修订、人工动作及冻结报告保留原记录。原持续预警依据失效时，新修订标为 `evidence_status=insufficient_after_late_arrival`，附带 `late_observation_refs` 和 `evidence_reassessment`。页面显示“补传后依据待核”，原等级作为历史记录，当前活跃预警及候选溯源按新依据显示。

## 专业预测与会话模型选择

`POST /api/forecasts` 完成预测后接着评估适用的物理浓度输出，并为新预测事件或等级升级提交简报。响应保留预测运行原字段，附带 `assessment/event_refs/briefs`。同等级重复计算沿用简报触发记录。

显式选择会话模型：

```http
POST /api/sessions/{session_id}/prediction-models
```

```json
{"point_id":"已登记测点","model_id":"已登记模型","version":"具体版本"}
```

服务校验当前来源、测点、模型及版本，保存到 `session.prediction_models[point_id]` 并记录动作审计。后续推进使用该绑定；缺少模型或输入返回具体缺项。未绑定点位沿用默认基线。关联重放复制父会话模型配置，运行、事件和时钟各自保存。

前端选择模型时保存会话绑定，恢复会话或切换测点时读取绑定；查看历史事件的关联预测继续显示其实际运行版本。

## 服务端连续监测

`REALTIME_MONITOR_ENABLED=true` 启用后台监测，`REALTIME_MONITOR_INTERVAL_SECONDS=5` 设置扫描间隔。后台只推进 `mode=realtime` 且处于 `running` 的会话；浏览器关闭后继续检查新观测、预测、事件与简报。历史回放由原回放控制推进。

同会话的推进、模型选择、人工预测和启停共用执行锁。暂停或停止后，后台在锁内重新核对状态；服务重启后从持久会话恢复扫描。专业模型在同一预测截止周期复用结果，实测数据每轮继续检查。单个会话失败独立记录错误并继续处理其他会话。

`GET /api/sessions/{session_id}` 的 `monitor` 返回启用状态、最近扫描、最近成功扫描、错误码及连续失败次数。扫描成功与模型就绪分别显示。停机先停止扫描，再等待在途计算和受管简报，最后关闭存储。

## 聊天恢复与工具查询

应用启动时，普通聊天的未完成任务改为 `failed/task_interrupted`，保留已完成步骤和结果，并结束未完成工具记录。用户可重新提交；系统恢复不自动重跑建议生成等有副作用的动作。

`POST /api/chat` 支持可选的 `point_id`，服务端校验其属于当前会话来源后绑定任务。页面提交当前选中测点，模型上下文同时保留该只读选择。

真实来源继续在本机调用业务工具：最近一小时监测、预警列表、指定事件依据、已审核资料检索、处置进展和指定模型预测。对象或参数不明确时返回 `awaiting_input`。页面与结果明确标记查询方式，保留实际工具步骤、结果卡及引用。

DeepSeek 是现有大模型接口。普通对话和主动简报继续共用 `EVENT_BRIEF_ALLOWED_SOURCE_IDS`；生产保持 `[]`，实际 DeepSeek 联调使用独立纯合成库。真实观测、知识原件、模型权重及本机路径保持本机处理。

明确业务查询须有本次成功工具结果。资料查询要求 `search_documents`，监测查询要求 `query_series`，事件、依据、处置及预测分别核对对应工具。模型先给出无依据正文时，服务丢弃该正文并最多追加一次指定工具的补查询，继续遵守原轮次、调用次数与时间预算。仍无结果时返回缺项；正文中的显式资料引用必须来自本次实际检索，未知编号不能标为完成答案。已完成结果卡继续保留。

## 知识材料入库

`app.scripts.import_knowledge_materials` 根据本机清单校验全部文件摘要、适用范围和版本，使用既有原件上传、解析与审核服务。重复运行复用已有文件和资料版本，写入前为现有 SQLite 运行库保存备份。

```powershell
Set-Location 'F:\煤矿瓦斯智能体\source\coal-mine-gas-risk-agent\backend'
& 'F:\煤矿瓦斯智能体\environments\backend\Scripts\python.exe' -m app.scripts.import_knowledge_materials `
  --manifest 'F:\煤矿瓦斯智能体\data\knowledge-materials\20261104\材料清单.json' `
  --database 'F:\煤矿瓦斯智能体\data\runtime\app.db' `
  --upload-dir 'F:\煤矿瓦斯智能体\data\runtime\knowledge-files' `
  --backup-dir 'F:\煤矿瓦斯智能体\outputs\knowledge-import\backup' `
  --actor-id 'user-authorized-knowledge-import' --face-id 31002 `
  --report 'F:\煤矿瓦斯智能体\outputs\knowledge-import\report.json'
```

默认导入为待审。授权审核人可在资料页核对原件、版本和适用范围后批准；命令行亦支持明确的 `--approve-material` 和 `--review-comment`。待核材料和研究草稿保持待审，批准时重新校验有效期。现有31002作业规程沿用用户确认暂作现行的版本。

本轮10份原件、1344个片段已登记到正式本机库。31002作业规程依据用户先前的现行版本确认启用检索，其余9份保持待审。具体版本、原件摘要及审核状态可在资料页和本机入库报告回查。验证产物位于 `F:\煤矿瓦斯智能体\outputs\reliability-local-20261104`，正式代码和合成测试同步 GitHub。
