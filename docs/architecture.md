# 模块职责、接口与调用关系

基础版采用一个FastAPI进程、SQLAlchemy存储和Vue工作台。监测值单独索引，其余业务记录按类型和编号存储；服务共享事务入口。当前以SQLite验收，PostgreSQL通过连接配置启用。

| 模块 | 职责 | 主要接口 | 调用 |
| --- | --- | --- | --- |
| auth | 账号、口令散列、令牌散列与工作面/角色授权 | login/get_actor/authorize/logout | 全部HTTP入口 |
| data | 数据源/点位、文件、历史导入、现场标准读数、质量及时钟查询 | import_history/ingest_readings/query_series | store |
| sessions | 会话模式、状态和单调数据时钟 | create_session/apply_action/get_context | data目录 |
| prediction | 不可变模型版本、闭合输入窗口、预测适配与运行记录 | register_model/run_prediction/get_forecast | data |
| alerts | 团队草案规则、实测/预测事件、不可变事件修订 | evaluate_observed/evaluate_forecast/query_events | data/预测结果 |
| traceability | 冻结事件对应的观测、模型和规则依据 | get_event_evidence | alerts/data/prediction |
| knowledge | 原文页段、审核、检索、建议内容版本、经验 | import_reference/search_documents/draft_advice | 服务端依据及批准引用 |
| cases | 人工确认、建议审核事实、执行/反馈/复核及冻结快照 | record_action/get_case/export_reviewed_case | alerts/knowledge |
| agent | DeepSeek八工具编排、任务、上下文隔离与审计 | chat/submit/get_task/cancel_task | 只读工具及建议草稿 |
| evaluation | 冻结清单、事件一对一匹配、分子分母和报告 | freeze_manifest/build_report/get_report | 冻结导出记录 |
| frontend | 会话、曲线、依据、资料及人工闭环、对话、评估 | /api/* | HTTP API |

`sessions/runner.py`组合一次时钟推进，计算观测预警和可用预测。相同会话、时钟、配置、模型清单及可见数据数量的重复推进复用结果。

```mermaid
flowchart LR
  UI[工作台] --> API[认证与HTTP校验]
  API --> S[会话与Runner]
  S --> D[监测数据]
  D --> P[预测]
  S --> A[规则与事件]
  P --> A
  A --> T[依据查询]
  T --> K[批准资料与建议版本]
  K --> C[人工审核及处置]
  C --> E[冻结经验快照]
  API --> G[DeepSeek工具任务]
  G --> D
  G --> T
  G --> K
  G --> C
  C --> V[冻结评估记录]
```

八工具为query_series、run_prediction、get_forecast、query_events、get_event_evidence、search_documents、draft_advice、get_case。身份、来源和数据时钟由服务端注入，工具参数通过禁止多余字段的JSON Schema校验；上限5轮、12次调用。检索原文作为资料输入，审计保留名称、参数、状态、耗时和出处。

查询要求event_time和available_at均不晚于as_of，范围[start_at,end_at)。模型读取连续有效、来源允许的闭合时间桶。基线明确标记，专业浓度规则只采用residual_corrected物理浓度输出。

事件记录revision_as_of逻辑时钟；人工操作同时记录真实occurred_at和数据data_as_of。建议内容保存在knowledge，审核事实保存在cases。经验由已复核快照生成，继承演练/现场性质；现场检索过滤演练经验。

运行服务的/openapi.json列出实际HTTP接口，包含上传、目录、标准现场接收及基础方案兼容路径。HTTP监测导入只接受上传编号，文档文件读取须由可信本地调用配置允许目录。

专业模型首轮训练、产物登记及六通道加工数据接入已完成；模型效果与聚类问题见[训练与聚类验算](training-and-clustering-audit-20261103.md)。后续集成项包括模型改进与真实成因标签、现场系统协议、矿井有效预案和经确认的单位/规则。

## 专业模型封装

`gas_forecast` 封装 FEDformer＋12阶段 LightGBM 残差模型；`gas_attribution` 封装六通道 KMeans/SVC 候选溯源。两者通过 `ModelRuntime` 在独立 Python 环境执行，受信任的本地目录清单绑定模型版本、来源及工作面。路径和运行命令由本地配置提供。

`GET /api/model-packages` 查看当前会话适用的模型及实验状态；`POST /api/model-package-runs` 提交准备好的特征；`GET /api/model-package-runs/{run_id}` 查询本会话运行。准备特征入口记录 `submitted_features`；常规专业预测由 `TrainedModelsPipeline` 构造同训练定义的特征，记录 `server_observations` 和输入行引用。因果候选只处理观测事件中达到模型阈值的目标时刻，输入截止于该时刻之前300秒；当前SVC自动调用关闭。

依据追溯继续使用 `traceability`；候选溯源的类别概率为聚类成员概率。输入合同、产物导出及调用命令见[模型封装说明](model-packaging.md)。
