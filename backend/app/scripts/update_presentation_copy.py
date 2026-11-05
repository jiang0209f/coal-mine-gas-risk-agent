"""Prepare or update the existing local presentation copy after a Git release."""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import tarfile

_BASE = Path("F:/煤矿瓦斯智能体")
_EXCLUDED = {".git", "node_modules", ".venv", "__pycache__", ".pytest_cache", "local_data"}


def _git(source, *args):
    return subprocess.check_output(["git", "-c", "safe.directory="+source.as_posix(), "-C", str(source), *args], stderr=subprocess.PIPE)


def _source_commit(source, expected):
    current = _git(source, "rev-parse", "HEAD").decode().strip()
    if expected:
        published = _git(source, "rev-parse", "--verify", expected+"^{commit}").decode().strip()
        if published != current:
            raise ValueError("正式源码HEAD与指定发布提交不一致")
    return current


def _within(path, root):
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError("副本目标路径须位于指定演示目录内")
    return resolved


def _presentation_aliases(demo_root, aliases_file=None):
    settings = aliases_file or demo_root/"presentation-settings.json"
    if settings.is_file():
        value = json.loads(settings.read_text(encoding="utf-8-sig"))
        aliases = value.get("aliases") if isinstance(value, dict) else value
    else:
        # Recover the established local build setting without publishing its
        # original identities as constants in the source repository.
        aliases = None
        assets = demo_root/"source/coal-mine-gas-risk-agent/frontend/dist/assets"
        for bundle in sorted(assets.glob("*.js")):
            for match in re.finditer(r"JSON\.parse\('(\[\[.*?\]\])'\)", bundle.read_text(encoding="utf-8")):
                try:
                    aliases = json.loads(match[1])
                except ValueError:
                    continue
                break
            if aliases:
                break
    if not isinstance(aliases, list) or not aliases or any(not isinstance(pair, list) or len(pair) != 2
            or any(not isinstance(item, str) or not item.strip() for item in pair) or pair[0] == pair[1] for pair in aliases):
        raise ValueError("未找到现有展示别名；请通过--aliases-file提供本机JSON配置")
    return aliases


def _copy_commit(source, destination, commit):
    copied = 0
    archive_bytes = _git(source, "archive", "--format=tar", commit)
    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or any(part in _EXCLUDED for part in relative.parts)
                    or (relative.name.startswith(".env") and relative.name != ".env.example")):
                continue
            target = _within(destination.joinpath(*relative.parts), destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream:
                with stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
                copied += 1
    return copied


def _backup(database, destination):
    if database.resolve() == destination.resolve():
        raise ValueError("正式数据库与副本数据库必须不同")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True) as source:
        with sqlite3.connect(destination) as target:
            target.execute("PRAGMA busy_timeout=5000")
            source.backup(target, pages=2048, sleep=0.01)


def _powershell(value):
    return "'"+str(value).replace("'", "''")+"'"


def _configuration(demo_root, source, runtime, aliases):
    # This independent file deliberately contains no provider credential.
    environment = {
        "RUNTIME_DIR": str(runtime), "DATABASE_URL": "sqlite+pysqlite:///"+(runtime/"app.db").as_posix(),
        "MODEL_CATALOG_PATH": str(_BASE/"models/model-catalog.json"),
        "KNOWLEDGE_EMBEDDING_MODEL_DIR": str(_BASE/"models/knowledge-embedding/bge-small-zh-v1.5/7999e1d3359715c523056ef9478215996d62a620"),
        "KNOWLEDGE_RERANKER_MODEL_DIR": str(_BASE/"models/knowledge-reranker/mmarco-mMiniLMv2-L12-H384-v1/1427fd652930e4ba29e8149678df786c240d8825"),
        "KNOWLEDGE_INDEX_DIR": str(demo_root/"knowledge-index"),
        "INGESTION_WORKER_ENABLED": "false", "REALTIME_MONITOR_ENABLED": "false",
        "CONTROL_WORKER_ENABLED": "false",
        "EVENT_BRIEF_ALLOWED_SOURCE_IDS": "[]", "AGENT_REAL_SOURCE_IDS": "[]", "AGENT_REAL_DATA_MODE": "disabled",
        "DEEPSEEK_API_KEY": "",
    }
    config_path = runtime/".env.presentation"
    config_path.write_text("\n".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in environment.items())+"\n", encoding="utf-8")
    launch_environment = {**environment, "COAL_AGENT_ENV_FILE": str(config_path), "PYTHONPATH": str(source/"backend"),
        "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TEMP": str(runtime/"tmp"), "TMP": str(runtime/"tmp")}
    output = _BASE/"outputs/presentation-copy-20261105"
    lines = ["$ErrorActionPreference='Stop'", "$sourceRoot="+_powershell(source), "$outputRoot="+_powershell(output),
        "if (Get-NetTCPConnection -LocalPort 8012 -State Listen -ErrorAction SilentlyContinue) { throw '8012端口已有服务，请在既定停机窗口启动更新副本。' }"]
    lines.extend(f"$env:{key}={_powershell(value)}" for key, value in launch_environment.items())
    lines += ["New-Item -ItemType Directory -Force -Path $env:TEMP,$outputRoot | Out-Null",
        "$process=Start-Process -FilePath "+_powershell(_BASE/"environments/backend/Scripts/python.exe")+
        " -ArgumentList '-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8012' -WorkingDirectory $sourceRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputRoot 'demo.stdout.log') -RedirectStandardError (Join-Path $outputRoot 'demo.stderr.log') -PassThru",
        "Write-Output ('demo_launcher_pid='+$process.Id)"]
    (demo_root/"启动演示.ps1").write_text("\n".join(lines)+"\n", encoding="utf-8-sig")
    (demo_root/"presentation-settings.json").write_text(json.dumps({"mode": "anonymous", "aliases": aliases}, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="预览同步计划；正式Git发布后显式--apply更新现有F盘展示副本")
    parser.add_argument("--published-source", type=Path, default=_BASE/"source/coal-mine-gas-risk-agent")
    parser.add_argument("--database", type=Path, default=_BASE/"data/runtime/app.db")
    parser.add_argument("--demo-root", type=Path, default=_BASE/"演示副本/匿名汇报版-20261105")
    parser.add_argument("--expected-commit", help="正式Git发布提交；--apply时必须填写，且正式源码HEAD须一致")
    parser.add_argument("--aliases-file", type=Path, help="可选的本机别名JSON；默认沿用副本设置或已有构建配置")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    published, database, demo_root = args.published_source.resolve(), args.database.resolve(), args.demo_root.resolve()
    if not demo_root.is_dir():
        raise ValueError("现有演示副本目录不存在；请核实路径，不自动新建或移动旧副本")
    destination = _within(demo_root/"source/coal-mine-gas-risk-agent", demo_root)
    runtime = _within(demo_root/"runtime", demo_root)
    if not runtime.is_dir() or not destination.is_dir() or not database.is_file():
        raise ValueError("既有source/runtime目录或正式数据库不完整")
    commit = _source_commit(published, args.expected_commit)
    aliases = _presentation_aliases(demo_root, args.aliases_file)
    plan = {"state": "planned", "published_commit": commit, "published_source": str(published), "source_destination": str(destination),
        "database_source": str(database), "database_destination": str(runtime/"app.db"), "presentation_mode": "anonymous",
        "copies_node_modules": False, "copies_credentials": False, "workers_enabled": False, "real_source_api_enabled": False,
        "processes_stopped": 0, "processes_started": 0, "existing_process_restart_required": True}
    if not args.apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if not args.expected_commit:
        raise ValueError("--apply需要显式--expected-commit，正式代码发布后再同步")
    node = shutil.which("node")
    vite = published/"frontend/node_modules/vite/bin/vite.js"
    if not node or not vite.is_file():
        raise ValueError("请复用正式F盘前端现有Node/Vite依赖，不复制node_modules或另行安装")
    _git(published, "diff", "--quiet", commit, "--", "frontend")
    _backup(database, runtime/"app.db")
    plan["copied_source_files"] = _copy_commit(published, destination, commit)
    _configuration(demo_root, destination, runtime, aliases)
    build_environment = {**os.environ, "VITE_PRESENTATION_MODE": "anonymous", "VITE_PRESENTATION_ALIASES": json.dumps(aliases, ensure_ascii=False),
        "TEMP": str(runtime/"tmp"), "TMP": str(runtime/"tmp")}
    (runtime/"tmp").mkdir(parents=True, exist_ok=True)
    subprocess.run([node, str(vite), "build", "--outDir", str(destination/"frontend/dist")],
                   cwd=published/"frontend", env=build_environment, check=True)
    plan.update(state="completed", updated_at=datetime.now(timezone.utc).isoformat())
    (demo_root/"source-revision.json").write_text(json.dumps({"git_commit": commit, "source_copy_synchronized": True, "presentation_mode": "anonymous"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (demo_root/"presentation-update.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
