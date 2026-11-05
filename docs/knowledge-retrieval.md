# 知识检索与公开条文解读

本机检索将关键词、标准号/条款号、领域同义词和中文向量召回组合排序。原文、页段、版本与引用编号保留，排序分数用于比较候选相关性。

## 调用关系

`KnowledgeService.search_documents`先按工作面权限、批准状态、当前有效期和资料版本筛选，再调用`retrieval.retrieve`。原文存在已登记分歧时，结果附带双方引用、适用条件和核验状态。

`LocalTextEmbeddings`使用本机BGE中文文本向量模型的ONNX文件与Tokenizer。模型原件来自BAAI固定版本，运行时不下载。长片段按512-token窗口、128-token重叠编码并汇总，缓存绑定模型/配置摘要、窗口策略和文本SHA。模型、缓存与测试产物放F盘；缓存只存文本摘要和向量，仍按本机资料管理。

第一阶段保留编号匹配、词组覆盖、同义词及向量相似度依据。有向量时词法与语义得分各占一半，标准号及条款号匹配仍优先。原混合通道保留纯向量0.60、弱词法与向量联合0.55的门槛。向量不可用时明确降级为词法检索。

### 可选本机相关性重排

`KNOWLEDGE_RERANKER_MODEL_DIR`配置已核验的本机MiniLM量化ONNX模型。未配置时沿用上述检索。重排使用`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`固定版本`1427fd652930e4ba29e8149678df786c240d8825`，原件、清单与Apache-2.0许可存F盘。模型只判断问题与条文的相关性，文本回答继续由DeepSeek API或本机业务工具提供。

1. 权限、审核状态、当前有效版本和公开用途先于两个本机模型筛选。标准号/条款号及明确的主扇/局扇、供电/供气对象约束继续生效。
2. 合并原混合检索前50段和未经过0.60门槛过滤的语义前50段，按原引用编号去重，最多100段进入相关性重排。
3. MiniLM使用原问题与完整父段配对。查询超过128词元时取前缀并记录截断；配对总长上限512词元，仅对正文滑窗、重叠128词元，覆盖末尾后取父段最高窗口分数。默认`batch_size=1`，每次推理一个窗口，避免动态量化随同批其他候选或padding改变同一pair的分数。批次和padding策略绑定模型身份，结果保留计划窗口数、实算窗口数和批次数。
4. 第一阶段名次与MiniLM名次按`1/(60+first_rank) + 1/(60+rerank_rank)`等权融合。原混合通道已准入者的`first_rank`使用原混合名次；新增纯语义候选使用dense名次；无向量时使用词法名次。编号匹配保持优先。新增纯语义候选还须具有本机识别的业务上下文，且MiniLM分数不低于0.05；这两个条件只约束扩池准入，原混合通道已准入的候选保留。
5. 重排失败或输出异常时告警并回到原混合结果，扩充候选不会从失败路径进入结果。

结果保留`reranker_status`、`reranking_mode`、`recall_path`、`semantic_rank`、`retrieval_rank`、`retrieval_rank_source`、`rerank_rank`、`fusion_method`及可展开的排序说明。`retrieval_rank_source`分别为`hybrid`、`semantic`或`lexical`；`semantic_rank`始终单独表示真实dense名次，无向量时为空。分数与名次表示相关性；原文、适用条件和人工审核用于确定条文能否支持具体措施。

行业问法使用同一组可检查的主题词，包括风路与通风、标气与调校、备用机与双机热备、补来的数据与断线补传、覆盖旧数与记录更正。逃生口语同时出现灾害和离开动作时才补充撤离主题；“电源”与“接通”同时出现时补充复电主题；“还原”同时具有数据库、备份、拷贝等数据语境时才补充备份恢复主题。扩展用于查找原条文，人工/自动、设备类别、否定、数值和操作顺序保留在原文中。

口语触发同时核对对象：备用机需要主机/电脑语境，补来的数据及更正需要测值/原始记录等对象，接通电源需要浓度/停电等条件，撤离口语需要正在遇险/发生灾害等情境。多义词和设备子串设独立回归负例。

第二轮补充新鲜风/迎头送风、仪器测值调整与调校记录、完整性与恢复时点、原数值变更审计、钻孔气体抽取达标等组合主题，仅扩展本机查询主题，保留原问题和原文。能源恢复未指明供给对象时并列检索供电、供气和压风分支，结果带`energy_supply_type_unspecified`；明确对象的问句保留对应分支。

## 召回评测

v1合成资料包含36个候选片段、12个主题、78道问题，原开发集与保留集各39道，资料、金标和拆分记录SHA。第二轮已使用全部78题定位问题和补充回归，因此v1全部作为开发回归集。真实资料的金标、问句、引用和逐题结果仅存F盘。

v2在本轮实验前独立冻结60段、120题的新基准，开发/保留各60题。每组含36道支持题、18道严格负例及6道现场事实背景探针，使用独立`manifest.json`。开发阶段只使用`dev.json`；独立保留集在完整算法与推理配置冻结后评测一次，结果用于验收。

第二轮固定结果如下。R@5表示支持题的登记依据进入前五条的比例；严格负例单独统计有候选返回的问题数。

| 数据集 | 基线R@5 | 本轮R@5 | 严格负例有候选（基线→本轮） |
|---|---:|---:|---:|
| v1开发回归：48道支持题 | 41/48 | 48/48 | 9/24→9/24 |
| v2开发：36道支持题 | 28/36 | 32/36 | 6/18→6/18 |
| v2独立保留：36道支持题 | 29/36 | 30/36 | 8/18→8/18 |

旧7个漏项均进入前五条。新独立保留集R@10、Top1和MRR也从0.80556升至0.83333；六道背景探针仍有四道返回相关背景。支持题Top5含困难干扰段的问题保持9/36，Top10由9/36增至10/36，Top1困难干扰为0。两方案各60题全部有效，无模型降级、网络或云调用。共同预热BGE语料及查询缓存后，CPU单题平均耗时从0.099秒增至1.036秒，P95从0.106秒增至1.321秒；正式语料较长时另需测量完整查询耗时。新保留集仍有6道支持题漏召回，解封后未调整算法、参数或金标。

评测分别统计支持段Recall@k、首条命中率、MRR、跨主题返回率、缺少依据的问题返回率，以及允许相关背景的事实探针。背景条文与现场事实分开计数。报告记录语料/问题/代码/模型摘要和逐题排名，使用本机模型并阻断网络。

逐题保留检索模式及向量状态。向量异常降级时，整份语义评测标记为无效并以非零状态退出，降级题单独列出；主要指标只计有效题，另保留全部问题的诊断指标。

v2开发回归运行示例（源码已同步到F盘，模型、索引和报告均指向F盘）：

```powershell
$projectRoot = 'F:\煤矿瓦斯智能体\source\coal-mine-gas-risk-agent'
$systemRevision = git -C $projectRoot rev-parse HEAD
Set-Location "$projectRoot\backend"
& 'F:\煤矿瓦斯智能体\environments\backend\Scripts\python.exe' -m app.scripts.evaluate_knowledge_retrieval `
  --dataset "$projectRoot\tests\fixtures\knowledge_retrieval_v2\dev.json" `
  --manifest "$projectRoot\tests\fixtures\knowledge_retrieval_v2\manifest.json" `
  --split dev --k 5 `
  --model-dir 'F:\煤矿瓦斯智能体\models\knowledge-embedding\bge-small-zh-v1.5\7999e1d3359715c523056ef9478215996d62a620' `
  --reranker-model-dir 'F:\煤矿瓦斯智能体\models\knowledge-reranker\mmarco-mMiniLMv2-L12-H384-v1\1427fd652930e4ba29e8149678df786c240d8825' `
  --index-dir 'F:\煤矿瓦斯智能体\outputs\knowledge-retrieval-dev\index' `
  --output 'F:\煤矿瓦斯智能体\outputs\knowledge-retrieval-dev\report.json' `
  --system-revision $systemRevision
```

报告记录两个模型的原件、运行身份和全部窗口统计。重排退回旧检索会使本次重排评测无效。

使用相同语料、模型和金标比较基线与改动。保留集解封后的结果用于最终回归，后续优化使用新的保留集。

## 两种资料范围

- `local_only`：默认范围，包括旧资料、矿井规程、现场数据、处置经验、图纸及未核OCR。批准和适用期符合时可在本机检索。
- `public_normative`：安全管理角色按具体资料版本登记官方HTTPS出处、类别和核验说明，策略绑定整份正文摘要。只有批准且当前有效、正文摘要一致的公开法规标准可用于云端解读。

新版本、撤销公开用途、失效和正文变化均须重新核验。仅在上传内容中写`disclosure`不会改变资料外发范围。

公开入口为`POST /api/references/public-answer`，请求只含`session_id`与`query`。原提问只在本机用于检索；DeepSeek请求使用固定解读题意、净化后的公开条文、官方出处、公开页段与临时PUB引用。现场执行上下文、旧对话、本机路径、矿井私有材料和原问题不进入该请求。返回引用须属于本次证据。

普通问答和事件简报继续使用原合成来源白名单，真实来源保持本机处理。旧云工具中的知识检索也执行公开范围筛选；本机历史及已撤回公开权限的引用不会作为后续云对话依据。

## 条文分歧

`POST /api/references/conflicts`由安全管理角色登记引用编号、主题、适用条件、原因和状态。数值不同本身不等于同一设备/场景下冲突。登记的分歧保留原文与双方位置；解决分歧须记录核验依据。

未解决分歧会阻断相应引用的建议草稿和既有建议审核，本机问答与资料页面显示待核状态。公开冲突的参与方和解决依据都须满足当前公开政策；任一资料撤回、失效或变化后，该冲突记录留在本机。

## 配置与接口

运行依赖：`requirements-knowledge-cpu-win.lock.txt`；一次性离线导出依赖：`training/requirements-knowledge-export-cpu-win.lock.txt`。

配置`KNOWLEDGE_EMBEDDING_MODEL_DIR`指向已核验`manifest.json`的模型目录，`KNOWLEDGE_INDEX_DIR`指向F盘索引目录。未配置向量模型时保留词法与同义词检索。

MiniLM仅通过显式下载命令获取固定官方版本，服务启动时只加载本机文件并校验SHA，不触发下载：

```powershell
Set-Location 'F:\煤矿瓦斯智能体\source\coal-mine-gas-risk-agent\backend'
& 'F:\煤矿瓦斯智能体\environments\backend\Scripts\python.exe' -m app.scripts.download_text_reranker `
  --model-dir 'F:\煤矿瓦斯智能体\models\knowledge-reranker\mmarco-mMiniLMv2-L12-H384-v1\1427fd652930e4ba29e8149678df786c240d8825' `
  --report-dir 'F:\煤矿瓦斯智能体\outputs\knowledge-reranker-download'
```

下载清单绑定官方仓库、固定revision、文件大小、SHA和Apache-2.0许可。运行配置示例：

```dotenv
KNOWLEDGE_EMBEDDING_MODEL_DIR=F:/煤矿瓦斯智能体/models/knowledge-embedding/bge-small-zh-v1.5/7999e1d3359715c523056ef9478215996d62a620
KNOWLEDGE_RERANKER_MODEL_DIR=F:/煤矿瓦斯智能体/models/knowledge-reranker/mmarco-mMiniLMv2-L12-H384-v1/1427fd652930e4ba29e8149678df786c240d8825
KNOWLEDGE_INDEX_DIR=F:/煤矿瓦斯智能体/data/knowledge-index/bge-small-zh-v1.5
```

两个检索入口共享同一个本机重排实例；`KNOWLEDGE_RERANKER_MODEL_DIR`留空时关闭MiniLM，保留第一阶段检索。

- `POST /api/references/search`：本机检索；`scope=public`仅检索已登记公开资料。
- `POST /api/references/{entry_id}/disclosure`：登记或撤销具体版本的公开用途。
- `POST /api/references/public-answer`：DeepSeek公开条文解读。
- `GET /api/references/fragments/{reference_id}`：按权限回查原引用。
- `POST /api/references/conflicts`：登记或更新分歧。

上述功能生成检索结果、条文解读和建议草稿；现场动作和设备控制沿用人工处置流程。
