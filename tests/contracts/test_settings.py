from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings


@pytest.fixture(autouse=True)
def isolate_deepseek_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_TIMEOUT_SECONDS",
        "DEEPSEEK_MAX_RETRIES",
        "DEEPSEEK_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_allow_startup_without_api_key() -> None:
    settings = Settings(_env_file=None)

    assert settings.deepseek_api_key is None
    assert str(settings.deepseek_base_url).rstrip("/") == "https://api.deepseek.com"
    assert settings.deepseek_model == "deepseek-flash"
    assert settings.deepseek_timeout_seconds > 0
    assert settings.deepseek_max_retries in (0, 1)
    assert settings.deepseek_max_tokens > 0


def test_api_key_is_secret_and_hidden_in_settings_representation() -> None:
    secret = "test-key-for-settings-only"
    settings = Settings(_env_file=None, deepseek_api_key=secret)

    assert isinstance(settings.deepseek_api_key, SecretStr)
    assert settings.deepseek_api_key.get_secret_value() == secret
    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert secret not in settings.model_dump_json()


def test_environment_overrides_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "DEEPSEEK_API_KEY=test-key-from-file\n"
        "DEEPSEEK_MODEL=file-model\n"
        "DEEPSEEK_TIMEOUT_SECONDS=12\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-from-environment")
    monkeypatch.setenv("DEEPSEEK_MODEL", "environment-model")

    settings = Settings(_env_file=dotenv_file)

    assert settings.deepseek_api_key.get_secret_value() == "test-key-from-environment"
    assert settings.deepseek_model == "environment-model"
    assert settings.deepseek_timeout_seconds == 12


def test_explicitly_disabling_dotenv_does_not_load_working_directory_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "DEEPSEEK_API_KEY=test-key-from-unselected-file\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None)

    assert settings.deepseek_api_key is None


def test_environment_values_are_converted_to_their_declared_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setenv("DEEPSEEK_MAX_RETRIES", "1")
    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS", "256")

    settings = Settings(_env_file=None)

    assert settings.deepseek_timeout_seconds == 7.5
    assert settings.deepseek_max_retries == 1
    assert settings.deepseek_max_tokens == 256


@pytest.mark.parametrize("timeout_seconds", [0, -1])
def test_nonpositive_timeout_is_rejected(timeout_seconds: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, deepseek_timeout_seconds=timeout_seconds)


@pytest.mark.parametrize("max_retries", [0, 1])
def test_retry_limit_accepts_zero_or_one(max_retries: int) -> None:
    settings = Settings(_env_file=None, deepseek_max_retries=max_retries)

    assert settings.deepseek_max_retries == max_retries


@pytest.mark.parametrize("max_retries", [-1, 2])
def test_retry_limit_rejects_out_of_range_values(max_retries: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, deepseek_max_retries=max_retries)


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_nonpositive_token_limit_is_rejected(max_tokens: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, deepseek_max_tokens=max_tokens)
