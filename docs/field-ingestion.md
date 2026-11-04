# 现场数据接入与补传

通用接入层接收 JSON 批次，完成字段适配、点位校验、持久保存、去重和运行状态查询。上游接口地址、协议、数据库结构与鉴权方式待现场资料提供后适配。本文模板均为合成记录，源时区为 `+08:00`，示例采样时间为 2026-01-01。

真实监测记录在本机适配、缓存和入库。环境、数据库、待发文件及运行产物放 F 盘。接入模块使用已有后端鉴权；当前阶段沿用现有页面，不新增登录页面。

## 1. 登记来源和点位

以下 HTTP 请求均使用具备数据管理权限、可访问目标工作面的本机 Bearer 令牌。先调用 `POST /api/sources`：

```json
{
  "source_id": "synthetic-ingestion-source",
  "name": "合成接入来源",
  "face_id": "31002",
  "timezone": "+08:00"
}
```

再调用 `POST /api/points`：

```json
{
  "point_id": "synthetic-ingestion-point",
  "source_id": "synthetic-ingestion-source",
  "source_tag": "SYNTH-T0",
  "name": "合成甲烷测点",
  "quantity": "methane",
  "unit": "%",
  "unit_status": "confirmed",
  "mapping_status": "confirmed",
  "lineage": "raw"
}
```

`face_id` 使用当前身份获准访问的工作面。接入器绑定一个来源，所有测点必须属于该来源。来源和点位元数据登记后保持固定，重复相同内容可复用，变更内容返回冲突。

## 2. 选择适配格式并登记接入器

### 标准 JSON

向 `POST /api/ingestion/connectors` 提交：

```json
{
  "connector_id": "synthetic-standard-connector",
  "name": "合成标准格式接入",
  "source_id": "synthetic-ingestion-source",
  "adapter_kind": "standard_json",
  "mapping": {"timezone": "+08:00"},
  "tag_map": {},
  "enabled": false,
  "stale_after_seconds": 120
}
```

标准批次的 `payload` 采用记录数组，或包含 `readings` 数组的对象。[标准数据模板](examples/ingestion-standard.json)可直接作为 CLI 的 `--payload-file`：

```json
{
  "readings": [
    {
      "reading_id": "synthetic-standard-20260101-0001",
      "point_id": "synthetic-ingestion-point",
      "event_time": "2026-01-01T08:00:00+08:00",
      "raw_value": 0.42
    }
  ]
}
```

### 字段映射 JSON

将[字段映射接入器模板](examples/ingestion-mapped-connector.json)完整提交给 `POST /api/ingestion/connectors`。该配置匹配以下 `payload`：

```json
{
  "data": {
    "rows": [
      {
        "record_id": "synthetic-mapped-20260101-0001",
        "tag": "SYNTH-T0",
        "collected_at": "2026-01-01 08:00:01",
        "value": 0.43,
        "sequence": 1
      }
    ]
  }
}
```

`records_path` 定位记录数组，空字符串或省略表示根数组；其他字段路径使用字典键组成的点路径，如 `measure.value`。`tag_field/time_field/value_field` 必填，`id_field/sequence_field` 按上游数据配置。`tag_map` 将上游标签绑定到已登记的同源 `point_id`。

源时间带时区时按其时区转换为 UTC；无时区时间使用配置中的时区，默认沿用来源时区，支持 `+08:00` 或 IANA 名称。晚于服务端接收时间的记录直接拒绝。每批为 1 至 10000 条，规范化序列化后的原始 JSON 上限为 8 MiB。原始数值文本和空值保留，入库时执行既有数据质量检查。

接入器登记后默认为停用。确认字段、点位和时区后，向 `POST /api/ingestion/connectors/{connector_id}/actions` 提交 `{"action":"enable"}`。提交 `{"action":"pause"}` 可暂停入库，暂停期间仍可接收并持久保存批次。

## 3. 提交批次并核对收据

`POST /api/ingestion/connectors/{connector_id}/batches` 的请求体为：

```json
{
  "batch_id": "synthetic-batch-20260101-0001",
  "payload": {
    "readings": [
      {
        "reading_id": "synthetic-standard-20260101-0001",
        "point_id": "synthetic-ingestion-point",
        "event_time": "2026-01-01T08:00:00+08:00",
        "raw_value": 0.42
      }
    ]
  }
}
```

接收端先保存规范化记录与配置版本，再返回收据，例如：

```json
{
  "connector_id": "synthetic-standard-connector",
  "batch_id": "synthetic-batch-20260101-0001",
  "durable": true,
  "status": "pending",
  "imported_rows": 0,
  "duplicate_rows": 0
}
```

`durable:true` 表示服务端已持久保存该批次。`pending/retry_wait` 表示等待入库或重试；`completed` 才表示批次处理完成，并给出 `applied_at`、新增条数和重复条数；`rejected` 表示需要核对数据。用 `GET /api/ingestion/connectors/{connector_id}/batches/{batch_id}` 查询最新结果，或用批次列表接口分页回查。

## 4. 双端缓存、去重与时间

| 位置 | 状态 | 含义与处理 |
| --- | --- | --- |
| 采集端 SQLite outbox | `pending` | 本机待发送；断线、超时、无效收据和临时 HTTP 错误后保留，按 1 至 60 秒退避 |
| 采集端 SQLite outbox | `sending` | 已领取发送租约；其他发送进程跳过，进程崩溃后 30 秒租约到期可恢复 |
| 采集端 SQLite outbox | `delivered` | 收到匹配接入器和批次编号的 2xx、`durable:true` 收据，服务端状态为 `pending/completed/retry_wait`；表示交付到服务端 |
| 采集端 SQLite outbox | `blocked` | 401、409、服务端拒绝或重定向等需人工处理；核对后运行 CLI `retry` |
| 服务端持久队列 | `pending/retry_wait` | 等待启用或后台入库；临时失败按 1 至 300 秒退避 |
| 服务端持久队列 | `completed/rejected` | 完成入库或拒绝；拒绝记录保留原因，核对后调用接入器动作 `retry` |

服务端 `retry` 将该接入器的 `retry_wait/rejected` 批次重新置为 `pending`，保持启停状态。客户端与服务端重试分别处理各自保存的队列。原始载荷和编号保持固定；冲突需核查源记录，原批次可按原内容重试。已入库测值的更正规则在具体监测系统接入时确定。

同一接入器的同一 `batch_id`、相同内容返回已有收据；内容不同返回 HTTP 409 `batch_conflict`。ACK 丢失后使用原编号和原载荷重发。同一来源、测点和 `reading_id` 的相同观测只入库一次；时间或原始值不同则整批回滚并标为 `reading_conflict`。缺少源记录 ID 时，适配器按来源、点位、UTC 采样时间和可选顺序号生成稳定 ID，数值与到达时间均不参与。一个测点同一时刻存在多条合法记录时，上游应提供稳定 ID 或顺序号。

`event_time` 始终保留采样时刻，`accepted_at` 记录服务端接收时刻，`applied_at` 记录成功入库时刻。观测的 `available_at` 使用成功入库时刻。查询同时要求 `event_time` 和 `available_at` 不晚于会话 `as_of`，因此补传数据从本次成功入库时刻起可见；回放不会提前看到后来补传的观测。

## 5. F 盘本机补传 CLI

在 F 盘运行副本的 `backend` 目录执行。缓存文件与载荷文件放在 F 盘，复制标准模板时保留其 JSON 内容。`--payload-file` 只含上游载荷，`batch_id` 由参数指定。

```powershell
Set-Location 'F:\煤矿瓦斯智能体\source\coal-mine-gas-risk-agent\backend'
$ingestionPython = 'F:\煤矿瓦斯智能体\environments\backend\Scripts\python.exe'
$ingestionDatabase = 'F:\煤矿瓦斯智能体\data\runtime\ingestion-outbox.db'

& $ingestionPython -m app.scripts.ingestion_bridge --database $ingestionDatabase enqueue --connector-id synthetic-standard-connector --batch-id synthetic-batch-20260101-0001 --payload-file 'F:\煤矿瓦斯智能体\source\coal-mine-gas-risk-agent\docs\examples\ingestion-standard.json'
& $ingestionPython -m app.scripts.ingestion_bridge --database $ingestionDatabase status

$ingestionCredential = Read-Host '输入本机后端授权令牌' -AsSecureString
$env:COAL_INGEST_TOKEN = [Net.NetworkCredential]::new('', $ingestionCredential).Password
& $ingestionPython -m app.scripts.ingestion_bridge --database $ingestionDatabase flush-once --base-url http://127.0.0.1:8000 --token-env COAL_INGEST_TOKEN
Remove-Item Env:\COAL_INGEST_TOKEN
```

持续补传使用同样的临时环境变量，运行 `run --base-url http://127.0.0.1:8000 --token-env COAL_INGEST_TOKEN --interval 2`，按 `Ctrl+C` 停止。`flush-once/run` 的 `--max-batches` 默认 10。阻塞批次使用 `retry --connector-id synthetic-standard-connector --batch-id synthetic-batch-20260101-0001` 恢复，再执行发送。

令牌仅从当前进程环境读取，SQLite、状态输出和命令行参数均不保存令牌值。客户端只允许 `http://localhost`、`http://127.0.0.1` 或 `http://[::1]` 的服务根地址及本机端口；拒绝 URL 内的凭据、查询参数、片段和路径，禁用环境代理与 HTTP 重定向。

## 6. 后台任务与运行状态

后端配置 `INGESTION_WORKER_ENABLED=true` 为默认值。服务启动后约每秒扫描一次持久队列，每轮最多处理 10 批，只处理已启用的接入器；重启后继续扫描已有队列。设置为 `false` 并重启后暂停后台扫描，批次仍可持久接收。

`GET /api/ingestion/status?source_id=synthetic-ingestion-source` 返回当前身份可见的接入器和 worker 状态。worker 包含 `enabled/state/last_run_at/last_success_at/last_error_code`：`running` 表示扫描正常，`degraded` 表示扫描异常；worker 的成功时间表示扫描执行成功，单批入库结果以批次收据为准。

接入器包含 `disabled/awaiting_data/receiving/stale/retry_wait/error` 状态，以及接收时间、入库成功时间、最新源采样时间、心跳时间、批次数和行数。`stale_after_seconds` 默认 120，按最后一次成功接收批次的时间判断接收中断。`POST /api/ingestion/connectors/{connector_id}/heartbeat` 更新心跳时间，心跳与收到监测记录分别计时。

现场联调前需提供：上游协议或数据库结构、鉴权与访问范围、字段示例、点位表、单位和源时区、采样间隔、稳定记录 ID/顺序号、历史补传或游标规则、连接恢复行为。取得这些资料后再实现相应采集适配并验证现场链路。
