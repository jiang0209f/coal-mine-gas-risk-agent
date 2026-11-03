"""由本机管理员建立账号，口令通过交互或私有文件读取。"""

import argparse
from getpass import getpass
import json
from pathlib import Path

from app.config import Settings
from app.db import Store
from app.schemas.common import DomainError
from app.services.auth.service import AuthService


def main(argv=None, settings=None):
    parser = argparse.ArgumentParser(description="建立本地登录账号")
    parser.add_argument("--username", required=True)
    parser.add_argument("--face-id", action="append", required=True)
    parser.add_argument("--role", action="append", choices=["safety_management", "ventilation", "gas_governance"])
    parser.add_argument("--password-file", type=Path)
    args = parser.parse_args(argv)
    settings = settings or Settings(_env_file=None)
    password = args.password_file.read_text(encoding="utf-8-sig").strip() if args.password_file else getpass("初始口令（至少12字符）: ")
    store = Store(settings.database_url or "sqlite+pysqlite:///" + (settings.runtime_dir.resolve() / "app.db").as_posix())
    try:
        user = AuthService(store).create_user(args.username, password, args.role or ["safety_management"], args.face_id)
        print(json.dumps({"status": "created", "user": user}, ensure_ascii=False))
        return 0
    except DomainError as error:
        print(json.dumps({"status": "failed", "code": error.code, "message": error.message}, ensure_ascii=False))
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
