# 专业预测与溯源模型封装

本轮提供两个 Python 模型适配器、独立进程 JSON 运行器，以及经认证的模型目录和运行记录接口。模型输入由调用方按训练清单准备，封装负责校验、加载固定产物、推理和保存结果。

原始资料包含算法源码和数据；本轮已新增真实训练权重，并用共享特征函数接通常规预测接口。原未训练模板保留，新增训练包位于 `models/trained`。专业预测为实验候选，现有SVC自动溯源已启用；训练数值和聚类验算见[训练与聚类验算](training-and-clustering-audit-20261103.md)。

## 1. 模块与调用关系

| 模块 | 职责 |
| --- | --- |
| `app.adapters.gas_forecast.GasForecastAdapter` | FEDformer 基础预测、十二个 LightGBM 残差模型、标准化还原及分阶段输出 |
| `app.adapters.gas_attribution.GasAttributionAdapter` | 六通道历史输入、旧版统计特征、固定 SVC 推理及候选原因映射 |
| `app.adapters.model_bundle.ModelBundle` | 读取清单，校验包内路径、产物格式及 SHA-256 |
| `app.scripts.model_worker` | 在指定 Python 环境中检查或推理；标准输入、标准输出使用 JSON |
| `app.adapters.model_runtime.ModelRuntime` | 从私有目录选择包和解释器，启动子进程，限制超时，检查返回身份 |
| `app.services.prediction.packages.ModelPackagesService` | 会话、角色、工作面及来源授权，输入时钟校验、运行保存和查询 |

调用链为：

```text
已登录调用方
  → 模型包 HTTP 接口
  → ModelPackagesService：检查会话、数据源、工作面和时钟
  → ModelRuntime：按私有目录启动独立进程
  → model_worker：读取 JSON
  → 对应适配器：校验清单、产物及准备好的特征
  → JSON 推理结果
  → 保存 model_package_run
```

准备特征入口用于直接检查模型。新增 `TrainedModelsPipeline` 从受权的六个测点读取原始行，使用共享五分钟特征函数，已衔接 `/api/forecasts` 和智能体 `run_prediction`。已训练专业模型设置 `default_eligible=false`，显式传入其编号调用。

运行记录标记 `input_origin="submitted_features"`，保存输入 SHA-256、特征名、模型版本和会话时钟。这些输入来自本次提交，不能作为原始监测文件或行号的自动取证记录。

## 2. F 盘目录与运行配置

```text
F:\煤矿瓦斯智能体\
  environments\models\                         模型专用 Python 环境
  models\model-catalog.json                     私有模型目录
  models\packages\forecast-31002\               预测包
  models\packages\attribution-31002\            溯源包
  source\coal-mine-gas-risk-agent\               项目源码
```

`model-catalog.json` 格式：

```json
{
  "python_executable": "F:/煤矿瓦斯智能体/environments/models/Scripts/python.exe",
  "timeout_seconds": 120,
  "models": [
    {
      "bundle_dir": "packages/forecast-31002",
      "source_id": "history-31002",
      "face_id": "31002"
    },
    {
      "bundle_dir": "packages/attribution-31002",
      "source_id": "history-31002",
      "face_id": "31002"
    }
  ]
}
```

`bundle_dir` 相对模型目录文件所在位置解析。`source_id`、`face_id` 必须与实际登记的数据源及会话一致；模型编号和版本取自各包的 `manifest.json`。

后端启动前设置可选配置：

```powershell
$env:MODEL_CATALOG_PATH='F:\煤矿瓦斯智能体\models\model-catalog.json'
```

未配置目录时，模型包列表为空。模型依赖安装在 `environments\models`，与 FastAPI 后端环境分开。运行器启动子进程时仅传递必要系统环境及项目代码路径，API 请求只能提交特征，不能指定解释器、代码目录或产物路径。

## 3. 预测包：FEDformer + LightGBM

本次 Windows CPU 环境使用 Python 3.12，安装版本记录在 `training/requirements-models-cpu-win.lock.txt`。该锁文件用于独立模型环境，后端继续使用原环境。原 FEDformer 源码导入 Matplotlib，模型环境已补齐此依赖。

### 输入与时间

- 输入为 **90 个连续的五分钟桶**，对应过去 **450 分钟**。
- `as_of` 是带时区的闭合五分钟桶边界。
- `times` 按时间升序表示桶起点，严格等于 `as_of - 450分钟` 至 `as_of - 5分钟`，步长为 300 秒。
- `values` 为 `90 × N` 矩阵，列名及顺序与清单 `feature_names` 一致；最后一列对应目标浓度。
- `residual_values` 为一个残差特征向量，顺序与 `residual_feature_names` 一致，分别交给十二个残差模型。
- 浓度单位为 `%`，`horizon_seconds` 可为 300～3600 秒的 300 秒整数倍，默认 3600 秒。

原数据加工以五分钟组的最后一分钟作为时间标记。本封装请求使用桶起点，通过清单中的 `architecture.time_feature_offset_seconds` 重建训练时间特征；对应原时间标记时设置为 **240 秒**。训练时区另由 `time_feature_timezone` 明确记录。

完整请求字段为：`as_of`、`unit`、`feature_names`、`times`、`values`、`residual_feature_names`、`residual_values`，以及可选的 `horizon_seconds`。特征须已经按训练过程加工完成，适配器再使用固定训练均值和尺度标准化。

### 产物清单

| 产物名称 | 内容 |
| --- | --- |
| `checkpoint` | 与清单架构匹配的 PyTorch `state_dict` |
| `scaler` | JSON：`feature_names`、训练期 `mean`、`scale` |
| `residual_01`～`residual_12` | 十二个 LightGBM Booster 文本模型，分别修正一个未来区间 |

每项产物均在 `manifest.json` 的 `artifacts` 中声明包内相对路径和 SHA-256。清单同时记录完整架构、随机种子、训练时间范围、训练源文件 SHA-256、单位、特征顺序，以及可信的 `legacy_source_root`。

架构使用原 Fourier FEDformer，实现要求 `seq_len=90`、`label_len=45`、`pred_len=12`、八个注意力头、`c_out=1`。随机 Fourier 模式选择与随机种子有关，重建网络时使用清单种子。推理加载固定训练产物，CPU 执行。

残差模型须声明 `residual_target="physical_error"`、`residual_selection_mode="v1"`，对应原正式训练脚本的“真实值减基础预测”。原脚本其他模式中，v3 会把阈值偏移加入残差目标，本封装拒绝将这类产物作为物理浓度残差加载。

### 输出

输出为未来 **十二个五分钟区间的浓度最大值预测**，每项包含 `start_at`、`end_at` 和 `value`，完整覆盖未来 60 分钟。

输出保留以下阶段：

1. `base_prediction`：FEDformer 输出按目标列训练尺度还原后的浓度预测。
2. `residual`：对应 LightGBM 模型给出的残差修正量。
3. `residual_corrected`：基础预测加残差修正后的浓度预测。
4. `decision_score`：清单配置阈值偏移时，额外输出用于决策的分值。

物理预测单位为 `%`。阈值偏移为 `true_threshold - best_model_threshold`，其结果单独标记为 `decision_score`、单位 `dimensionless`，与物理浓度分开保存。

直接调用方式：

```python
from app.adapters.gas_forecast import GasForecastAdapter

adapter = GasForecastAdapter(bundle_dir)
state = adapter.inspect()
result = adapter.predict(prepared_request)
```

预测包由训练端输出 `state_dict`、训练标准化 JSON 和十二个 Booster，再填写清单与文件哈希。本轮提供加载和推理接口。

## 4. 溯源包：六通道 SVC 候选分类

### 输入与保留的旧版特征

六路特征名称及顺序固定为：

```text
T0瓦斯浓度、风速、采煤机速度、采煤机瓦斯浓度、左截割电机电流、右截割电机电流
```

请求包含 `as_of`、`feature_names`、`times` 和 `values`。输入为 **61 × 6**，时刻由旧到新、严格间隔 60 秒，首尾相距 3600 秒，最后时刻等于 `as_of`。提前量由模型包固定，为 300～3600 秒的 300 秒整数倍。

`feature_version="legacy_row_stats_v1"` 保留原实现的统计轴：将时间顺序倒转，取最近六行，分别对每行的六个通道计算均值、总体方差、最大值和最小值，得到 24 维特征。该版本没有改为“六个通道各自沿整个历史窗统计”。

原训练代码对输入取绝对值，原 `predict.py` 对历史输入没有取绝对值。封装要求通过 `preprocessing="identity"` 或 `"absolute"` 明确声明实际产物的预处理约定。

原始历史源仍保留T0。新增 `multisensor-31002-2022` 加工历史源已包含六路输入，训练包绑定这一来源；不同来源不拼接。输入不齐时返回 `not_ready`。当前训练包的 `automatic_attribution=true`，查询实测事件依据时自动调用，响应中的 `candidate_attribution` 保存模型版本、目标观测、输入截止时间和逐行引用。

登记脚本参数 `--automatic-attribution` 启用、`--no-automatic-attribution` 关闭；省略参数保留已有设置。初次登记且没有设置时默认关闭。同一事件修订与模型配置的重复查询复用结果，状态变更后按新配置执行。配置由可信本地登记流程修改。

### 固定产物与输出

`classifier.skops` 保存已训练的 `sklearn.svm.SVC`，参数为 `kernel="rbf"`、`C=100`、`gamma=0.01`、`probability=True`，输入为 24 维，类别编号为 0～4。

`training.json` 保存：

- `cause_map`：聚类编号字符串到候选原因文字列表的映射。
- `cluster_centers`：五个六维聚类中心。
- `scaler`：六维 `data_min`、`data_max`、`scale`、`min`。
- `source_hashes`：源文件名称与 SHA-256。
- `training_range`：带时区的 `start_at`、`end_at`。

聚类中心和缩放器用于保留训练依据。纯推理调用固定 SVC，不重新聚类、训练或拟合缩放器。SVC 的 24 维统计输入沿用原实现，不另外进行六维聚类缩放。

输出通过 `classifier.classes_` 对应概率列，将类别按概率排序，按最大相邻概率差选择候选类，返回：

- `cluster_probabilities`、`selected_cluster_ids`。
- 去重后的 `candidate_causes`。
- 模型版本、特征版本、预处理、提前量、预测时钟与训练范围。
- `causal_status="candidate_attribution"`、`probability_semantics="cluster_membership"`。

这些概率表示 SVC 对聚类类别的分类输出，候选原因来自原聚类中心规则映射。真实成因核查需要现场证据。

直接调用与导出已训练模型：

```python
from app.adapters.gas_attribution import GasAttributionAdapter, export_bundle

adapter = GasAttributionAdapter(bundle_dir)
state = adapter.inspect()
result = adapter.predict(prepared_request)

manifest = export_bundle(
    classifier, cause_map, training_metadata, output_dir,
    model_id=model_id, version=version,
    horizon_seconds=300, preprocessing="identity",
)
```

`export_bundle` 校验已训练 SVC、训练元数据和类别映射，写出 skops、训练 JSON 和哈希清单；目录须为空，已有版本保持不变。

### 原代码问题的处理

本轮保留旧版统计轴并显式声明预处理。原 `cur_minmax` 在每个查询中追加数据并重新拟合缩放器的问题，由固定训练产物和纯推理入口隔离；概率列编号使用 `classes_` 映射。

后续若修正统计轴、时间窗或训练标签生成流程，应使用新特征版本并重新训练评估。原溯源报告的参照标签来自聚类和原因规则，其准确率不能替代现场真实成因标签验证。

## 5. 独立进程命令

在 PowerShell 中设置本地源码路径和模型解释器：

```powershell
$modelPython='F:\煤矿瓦斯智能体\environments\models\Scripts\python.exe'
$env:PYTHONPATH='F:\煤矿瓦斯智能体\source\coal-mine-gas-risk-agent\backend'
$env:PYTHONUTF8='1'
$OutputEncoding=[System.Text.UTF8Encoding]::new($false)
```

检查两个私有包：

```powershell
& $modelPython -m app.scripts.model_worker --kind gas_forecast --bundle 'F:\煤矿瓦斯智能体\models\packages\forecast-31002' --inspect
& $modelPython -m app.scripts.model_worker --kind gas_attribution --bundle 'F:\煤矿瓦斯智能体\models\packages\attribution-31002' --inspect
```

准备好本地 JSON 特征文件后，通过标准输入推理：

```powershell
Get-Content -LiteralPath 'F:\煤矿瓦斯智能体\data\prepared\forecast-request.json' -Raw -Encoding UTF8 |
  & $modelPython -m app.scripts.model_worker --kind gas_forecast --bundle 'F:\煤矿瓦斯智能体\models\packages\forecast-31002'

Get-Content -LiteralPath 'F:\煤矿瓦斯智能体\data\prepared\attribution-request.json' -Raw -Encoding UTF8 |
  & $modelPython -m app.scripts.model_worker --kind gas_attribution --bundle 'F:\煤矿瓦斯智能体\models\packages\attribution-31002'
```

输入文件应满足前述矩阵与时间合同。运行器仅向标准输出写一份 JSON，状态为检查时的 `ready/not_ready/failed`，或推理时的 `completed/not_ready/failed`。输入上限为 8 MiB；模型目录可配置 1～300 秒的子进程超时。

## 6. 经认证的 HTTP 接口

| 方法与路径 | 参数 | 作用 |
| --- | --- | --- |
| `GET /api/model-packages` | 查询参数 `session_id` | 列出当前会话来源、工作面可用的模型版本和就绪状态 |
| `POST /api/model-package-runs` | JSON：`session_id`、`model_id`、可选 `version`、`features` | 执行指定模型包并保存结果 |
| `GET /api/model-package-runs/{run_id}` | 查询参数 `session_id` | 在原会话范围内查询运行记录 |

接口使用现有 Bearer 登录令牌，允许通风、瓦斯治理和安全管理角色。服务端使用会话 `as_of`，拒绝不同的提交时钟或未来输入时刻。模型目录路径和解释器由本机配置提供。

PowerShell 调用示意，`$token`、`$sessionId`、`$modelId`、`$modelVersion` 使用登录和目录接口取得的值：

```powershell
$headers=@{Authorization="Bearer $token"}
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/model-packages?session_id=$sessionId" -Headers $headers

$prepared=Get-Content -LiteralPath 'F:\煤矿瓦斯智能体\data\prepared\forecast-request.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$runBody=@{session_id=$sessionId; model_id=$modelId; version=$modelVersion; features=$prepared} | ConvertTo-Json -Depth 100
$run=Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/model-package-runs' -Headers $headers -ContentType 'application/json; charset=utf-8' -Body $runBody
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/model-package-runs/$($run.run_id)?session_id=$sessionId" -Headers $headers
```

## 7. 验收口径与产物加载

软件验收覆盖输入矩阵、特征顺序、时区与连续性、训练截止时间、产物哈希、类别映射、概率和输出物理量、目录范围授权、会话隔离及运行器失败状态。

真实依赖冒烟使用合成数据训练的小型 SVC，以及随机权重的 FEDformer 和小型 Booster，检查安全导出、加载、真实依赖推理与 JSON 进程往返。该验收验证封装及运行条件；预测误差、提前预警能力和溯源效果需在真实训练产物及独立数据上评估。

本次后端完整回归555项通过、3项跳过；独立模型环境82项验收全部通过。报告保存在F盘 `outputs/model-packaging`。原始模型Python源码哈希已在 `models/source-inventory.json` 记录，验收后核对保持一致。

LightGBM 通过 Python 读取 UTF-8 模型文本，再使用 `Booster(model_str=...)` 加载，支持本机F盘中文目录。参数定义见 [LightGBM 官方文档](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)。

PyTorch 仅加载 `state_dict`，使用 `torch.load(..., map_location="cpu", weights_only=True)`，加载方式见 [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.load.html)。

SVC 使用 skops：先检查未知类型，拒绝未批准类型，再以 `trusted=[]` 加载并限定为 SVC；方法见 [skops 安全持久化文档](https://skops.readthedocs.io/en/stable/persistence.html)。原始数据、私有目录、环境及模型产物保存于 F 盘，GitHub 同步源码、测试和文档。
