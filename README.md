# 采煤工作面瓦斯超限预测及溯源管控智能体

当前开发阶段为工程基础与DeepSeek接入验证：提供本地健康接口、配置读取、DeepSeek聊天适配器和工具调用诊断。业务工具、数据接入与预测功能按整体方案逐步接入。

## 安装

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -e ".[dev]"
```

本次验证的依赖版本记录在`requirements-dev.lock.txt`。需要复现该环境时，先安装此文件中的依赖，再执行`pip install --no-deps -e .`。

## 配置DeepSeek

首次创建本地配置：

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

在`.env`填写`DEEPSEEK_API_KEY`。配置也可通过同名环境变量提供，环境变量优先于`.env`。默认模型名为`deepseek-flash`，地址为`https://api.deepseek.com`，可按账号可用模型修改。接入格式见[DeepSeek官方文档](https://api-docs.deepseek.com/)。

`.env`、本地数据和模型产物由`.gitignore`排除。程序只报告密钥是否配置，错误输出不包含密钥和上游响应正文。

## 验证

离线测试使用模拟HTTP传输，不调用外部API：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

配置Key后，运行真实接口诊断：

```powershell
.\.venv\Scripts\python.exe -m app.scripts.check_deepseek
```

诊断依次验证一次普通对话、一次`integration_echo`工具请求及回传工具结果后的答复，正常流程共3次模型请求。该工具只用于本地诊断，后续业务智能体使用方案中的八个工具。未配置Key时命令返回配置提示和非零退出码。

## 启动本地后端

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开`http://127.0.0.1:8000/health`检查后端。该接口只报告进程和本地配置状态，不检测DeepSeek网络连通性。正式业务对话接口在agent模块阶段接入。

## 目录

- `backend/app/config.py`：环境配置。
- `backend/app/adapters/deepseek.py`：模型请求、返回值检查和重试。
- `backend/app/scripts/check_deepseek.py`：本地连接与工具调用诊断。
- `backend/app/main.py`：FastAPI入口和健康接口。
- `backend/app/services/`：后续业务模块。
- `tests/`：配置、协议边界和流程验证。
