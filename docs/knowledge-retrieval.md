# 知识检索与公开条文解读

本机检索将关键词、标准号/条款号、领域同义词和中文向量召回组合排序。原文、页段、版本与引用编号保留，排序分数用于比较候选相关性。

## 调用关系

`KnowledgeService.search_documents`先按工作面权限、批准状态、当前有效期和资料版本筛选，再调用`retrieval.retrieve`。原文存在已登记分歧时，结果附带双方引用、适用条件和核验状态。

`LocalTextEmbeddings`使用本机BGE中文文本向量模型的ONNX文件与Tokenizer。模型原件来自BAAI固定版本，运行时不下载。长片段按512-token窗口、128-token重叠编码并汇总，缓存绑定模型/配置摘要、窗口策略和文本SHA。模型、缓存与测试产物放F盘；缓存只存文本摘要和向量，仍按本机资料管理。

重排保留编号匹配、词组覆盖、同义词及向量相似度依据。向量不可用时明确降级为词法检索。初始门槛为纯向量0.60、弱词法与向量联合0.55，后续依据标注检索集校准。

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

- `POST /api/references/search`：本机检索；`scope=public`仅检索已登记公开资料。
- `POST /api/references/{entry_id}/disclosure`：登记或撤销具体版本的公开用途。
- `POST /api/references/public-answer`：DeepSeek公开条文解读。
- `GET /api/references/fragments/{reference_id}`：按权限回查原引用。
- `POST /api/references/conflicts`：登记或更新分歧。

上述功能生成检索结果、条文解读和建议草稿；现场动作和设备控制沿用人工处置流程。
