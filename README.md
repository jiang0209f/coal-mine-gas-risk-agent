# 采煤工作面瓦斯风险智能体

提供历史回放、模拟数据流和现场标准数据接收，完成监测导入、预警、依据查询、资料检索、建议草拟、人工处置和评估。后端 FastAPI + SQLAlchemy，前端 Vue 3 + ECharts。真实来源在本机处理，DeepSeek仅通过业务工具读取纯合成隔离数据。

专业模型通过适配器接入；当前内置持久性基线用于检查软件流程，缺少模型或有效输入返回 `not_ready`。预警规则标记为团队草案，物理浓度与决策分值分别保存。详见[模块与接口](docs/architecture.md)。

FEDformer＋LightGBM 预测和 KMeans/SVC 候选溯源已用31002六通道加工历史数据完成首轮训练。专业预测接通常规预测接口，持久性保持默认，专业模型可显式调用。现有溯源已启用：查询实测预警依据时自动构造六通道输入，返回候选原因，同一事件修订复用运行记录。环境、权重和逐例报告位于F盘。详见[封装调用说明](docs/model-packaging.md)及[训练与聚类验算](docs/training-and-clustering-audit-20261103.md)。

后续模块职责、接口、调用关系与竞赛展示顺序见[整体模块设计 v0.6](docs/module-design-v0.6.md)。当前先贯通功能与页面联动，模型效果改进单独安排。

正式深蓝工作台已接通真实监测、指定版本预测和事件溯源，支持恢复已保存预测、查看12个区间明细及切换事件依据。操作与真实数据验收见[前端联动说明](docs/frontend-linkage.md)。

新增预警或等级升级后提交[主动研判任务](docs/event-briefs.md)，展示工具进度、冻结依据和引用。真实来源保留本机依据；纯合成隔离来源可生成DeepSeek研判简报。`EVENT_BRIEF_ALLOWED_SOURCE_IDS` 为普通对话和主动简报共用的合成来源允许名单，生产保持 `[]`。详见[数据使用规则](docs/event-brief-cloud-fields.md)。

[本地知识库与人工处置闭环](docs/knowledge-cases.md)已贯通原件上传、审核检索、建议引用、执行反馈、复核归档及经验审核。资料按当前有效、最新审核版本使用，建议分别记录监测`data_as_of`和资料`knowledge_as_of`。10份知识库材料原件、定位文本和标注模板保存在F盘，正式引用按业务审核结果确定。

## 安装与启动

Python 3.11+、Node.js 22+：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -e ".[dev]"
Copy-Item .env.example .env
cd frontend
npm ci
npm run build
cd ..
.\.venv\Scripts\python.exe -m app.scripts.bootstrap_user --username manager --face-id 31002
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

初始化时交互输入至少12字符的密码。打开[工作台](http://127.0.0.1:8000/)登录；[接口文档](http://127.0.0.1:8000/docs)提供实际OpenAPI定义。前端开发可运行 `npm run dev`，代理后端8000端口。

配置 `.env` 的 `DEEPSEEK_API_KEY`。普通对话在缺少Key或来源未列入合成允许名单时，基于本机事件查询返回 `structured_template`；未允许来源同时记录 `local_reason=cloud_source_not_authorized`。系统尚未部署本地大模型。模型连接诊断：`python -m app.scripts.check_deepseek`，仅发送固定测试文本和随机数。模型名和地址可配置，协议见[DeepSeek官方文档](https://api-docs.deepseek.com/)。

`RUNTIME_DIR`指定数据目录，默认 `local_data`；`DATABASE_URL`留空使用该目录的SQLite文件，也支持PostgreSQL连接。当前集成验收使用SQLite。`COAL_AGENT_ENV_FILE`可在启动进程时指定单独保存的私有配置。环境、密钥、原始数据和模型产物由Git排除。

## 数据接入

安全管理账号通过 `/api/sources`、`/api/points`显式登记工作面和点位，上传CSV/XLSX后提交 `/api/history-imports`。映射明确指定时间列、测值列、工作表、表头行、时区和加工来源。

系统保留原始值、行号、文件SHA-256、事件时间和可用时间。非法测值转为空值并标记，零值和高浓度保留。查询同时按事件时间、可用时间和会话时钟过滤；聚合只返回闭合桶，填补使用过去有效值。

```powershell
python -m app.scripts.import_history --path "数据.xlsx" --source-id history-31002 --face-id 31002 --point-id 31002:T0 --source-tag T0_044A02 --start 2022-01-01 --end 2023-01-01
```

现场系统提交到已认证的 `POST /api/ingest`：`{source_id,readings:[{reading_id,point_id,event_time,raw_value}]}`。时间须带时区，接收时间由服务端记录，稳定的reading_id用于重试去重。现场专有协议由适配器转换为该格式。

## 工作流

1. 选择数据源和点位，创建或恢复会话。
2. 推进会话调用数据、预测和预警服务。事件保留不可变修订、依据引用及标校质量标记。
3. 查询依据，检索已批准资料，生成带出处的建议草稿。
4. 用户依次确认事件、审核建议、记录执行、提交反馈和复核。历史及模拟操作标记为演练。
5. 对话任务可异步提交、轮询、取消并查看工具审计。人工审核和处置通过业务接口执行。
6. 冻结评估记录，输出事件一对一匹配、分子分母，以及包含和剔除疑似标校的两套口径。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest
cd frontend
npm test
npm run build
npm run test:e2e
```

后端离线测试使用合成数据与模拟模型。前端E2E使用隔离接口夹具和无头Edge/Chrome，可通过 `UI_TEST_BROWSER`指定浏览器。真实数据报告和私有运行记录保存在本地。GitHub Actions检查后端测试、前端测试和构建。
