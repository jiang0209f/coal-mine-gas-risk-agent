"""本地配置与环境变量。"""

from pathlib import Path
import os

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


__all__ = ["Settings"]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(os.environ.get("COAL_AGENT_ENV_FILE", PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

    runtime_dir: Path = PROJECT_ROOT / "local_data"
    database_url: str | None = None
    model_catalog_path: Path | None = None
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: AnyHttpUrl = "https://api.deepseek.com"
    deepseek_model: str = Field(default="deepseek-flash", min_length=1)
    deepseek_timeout_seconds: float = Field(default=30, gt=0)
    deepseek_max_retries: int = Field(default=1, ge=0, le=1)
    deepseek_max_tokens: int = Field(default=512, gt=0)

    @field_validator("deepseek_api_key")
    @classmethod
    def normalize_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        key = value.get_secret_value().strip()
        return SecretStr(key) if key else None

    @property
    def deepseek_configured(self) -> bool:
        return self.deepseek_api_key is not None
