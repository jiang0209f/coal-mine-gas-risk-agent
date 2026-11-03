import pytest

from app.config import Settings
from app.db import Store
from app.scripts.bootstrap_user import main
from app.services.auth.service import AuthService


def test_bootstrap_reads_password_privately_and_no_default_account(tmp_path, capsys):
    secret = tmp_path / "password.txt"
    secret.write_text("local-test-password-12345", encoding="utf-8")
    url = "sqlite:///" + (tmp_path / "app.db").as_posix()
    config = Settings(_env_file=None, runtime_dir=tmp_path, database_url=url)
    assert main(["--username", "manager", "--face-id", "31002", "--password-file", str(secret)], config) == 0
    assert "local-test-password" not in capsys.readouterr().out
    store = Store(url)
    assert AuthService(store).login("manager", "local-test-password-12345")["user"]["roles"] == ["safety_management"]
    assert main(["--username", "manager", "--face-id", "31002", "--password-file", str(secret)], config) == 1
    store.close()
