"""Command-line entry point for the local durable ingestion sender."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import time

from app.schemas.common import DomainError
from app.services.data.outbox import DurableOutbox


def _parser():
    parser = argparse.ArgumentParser(description="本机现场数据持久缓存与补传")
    parser.add_argument("--database", required=True, help="F 盘本机 SQLite 缓存文件")
    commands = parser.add_subparsers(dest="command", required=True)
    enqueue = commands.add_parser("enqueue", help="将上游 JSON 批次写入本机缓存")
    enqueue.add_argument("--connector-id", required=True)
    enqueue.add_argument("--batch-id", required=True)
    enqueue.add_argument("--payload-file", required=True)
    commands.add_parser("status", help="查看数量、批次状态和错误代码")
    retry = commands.add_parser("retry", help="手动恢复阻塞批次")
    retry.add_argument("--connector-id", required=True)
    retry.add_argument("--batch-id", required=True)
    for name in ("flush-once", "run"):
        sender = commands.add_parser(name, help="向本机服务补传缓存批次")
        sender.add_argument("--base-url", default="http://127.0.0.1:8000")
        sender.add_argument("--token-env", default="COAL_INGEST_TOKEN", help="保存令牌的环境变量名")
        sender.add_argument("--max-batches", type=int, default=10)
        if name == "run":
            sender.add_argument("--interval", type=float, default=2, help="补传轮次间隔秒数，1 至 60")
    return parser


def _emit(result):
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        box = DurableOutbox(args.database)
        if args.command == "enqueue":
            path = Path(args.payload_file)
            if path.stat().st_size > 8 * 1024 * 1024:
                raise DomainError("待发批次超过 8 MiB", code="payload_too_large", status_code=413)
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (ValueError, UnicodeError):
                raise DomainError("待发文件必须为有效 UTF-8 JSON", code="invalid_payload") from None
            _emit(box.enqueue(args.connector_id, args.batch_id, payload))
        elif args.command == "status":
            _emit({**box.status(), "batches": box.list_batches()})
        elif args.command == "retry":
            _emit(box.retry(args.connector_id, args.batch_id))
        else:
            if args.command == "run" and not 1 <= args.interval <= 60:
                raise DomainError("补传轮次间隔应为 1 至 60 秒")
            while True:
                _emit(box.flush_once(args.base_url, os.environ.get(args.token_env), args.max_batches))
                if args.command == "flush-once":
                    break
                time.sleep(args.interval)
        return 0
    except DomainError as error:
        _emit({"error": error.code, "message": error.message})
        return 2
    except (OSError, sqlite3.Error):
        _emit({"error": "local_storage_error", "message": "本机缓存或待发文件读写失败"})
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
