"""Persist DeepSeek settings locally and test only on an explicit request."""
import json
from pathlib import Path
from threading import RLock
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import AnyHttpUrl, SecretStr

from app.adapters.deepseek import DeepSeekAdapter, DeepSeekRequestError
from app.schemas.common import DomainError, now_utc, utcstr
from app.services.agent.disclosure import record_disclosure


class ModelConnectionService:
    def __init__(self, store, settings, adapter_factory=DeepSeekAdapter):
        self.store, self.settings, self.adapter_factory = store, settings, adapter_factory
        self.path = Path(settings.runtime_dir) / "model-connection.json"
        self.lock, self.source, self.warning = RLock(), "environment", None
        if self.path.is_file():
            try:
                if self.path.stat().st_size > 32768:
                    raise ValueError("configuration size")
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                self._apply(self._validate(saved["provider"], saved["base_url"], saved["model"], saved.get("api_key")))
                self.source = "saved"
            except (OSError, ValueError, KeyError, TypeError, DomainError):
                self.warning = "已保存配置无法读取，当前使用环境配置。"

    @staticmethod
    def _authorize(actor):
        actor.require_role("safety_management")
        if actor.is_system:
            raise DomainError("模型连接设置需要人工操作", "forbidden", 403)

    @staticmethod
    def _validate(provider, base_url, model, api_key):
        if provider != "deepseek":
            raise DomainError("本地接口为预留选项，当前仅启用DeepSeek API。", "not_implemented", 400)
        try:
            parts = urlsplit(base_url.strip())
            if (parts.scheme != "https" or parts.netloc.lower() != "api.deepseek.com"
                    or parts.query or parts.fragment or parts.path.rstrip("/") not in ("", "/v1")):
                raise ValueError("endpoint")
        except (AttributeError, TypeError, ValueError):
            raise DomainError("请填写DeepSeek官方HTTPS接口地址。", "invalid_model_endpoint") from None
        if (not isinstance(model, str) or not model.strip() or len(model) > 128
                or any(ord(char) < 32 for char in model)):
            raise DomainError("模型名需要1至128个可见字符。", "invalid_model_name")
        if isinstance(api_key, SecretStr):
            api_key = api_key.get_secret_value()
        if api_key is not None:
            if not isinstance(api_key, str) or len(api_key) > 4096 or any(char.isspace() for char in api_key.strip()):
                raise DomainError("API Key格式无效。", "invalid_api_key")
            api_key = api_key.strip() or None
        return dict(provider="deepseek", base_url="https://api.deepseek.com" + parts.path.rstrip("/"),
                    model=model.strip(), api_key=api_key)

    def _apply(self, config):
        self.settings.deepseek_base_url = AnyHttpUrl(config["base_url"])
        self.settings.deepseek_model = config["model"]
        self.settings.deepseek_api_key = SecretStr(config["api_key"]) if config["api_key"] else None

    def get(self, actor):
        self._authorize(actor)
        result = dict(provider="deepseek", base_url=str(self.settings.deepseek_base_url).rstrip("/"),
            model=self.settings.deepseek_model, configured=self.settings.deepseek_configured,
            source=self.source, local_available=False)
        if self.warning:
            result["warning"] = self.warning
        return result

    def save(self, actor, *, provider="deepseek", base_url, model, api_key=None):
        self._authorize(actor)
        with self.lock:
            config = self._validate(provider, base_url, model, api_key)
            if config["api_key"] is None and self.settings.deepseek_api_key is not None:
                config["api_key"] = self.settings.deepseek_api_key.get_secret_value()
            temporary = self.path.with_name(self.path.name + "." + uuid4().hex + ".tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with temporary.open("x", encoding="utf-8") as output:
                    json.dump(config, output, ensure_ascii=False)
                temporary.replace(self.path)
            except OSError:
                temporary.unlink(missing_ok=True)
                raise DomainError("连接配置保存失败，请检查本机配置目录。", "model_config_write_failed", 503) from None
            self._apply(config)
            self.source, self.warning = "saved", None
            self.store.put("model_connection_audit", str(uuid4()), dict(action="save", actor_id=actor.user_id,
                provider="deepseek", base_url=config["base_url"], model=config["model"], configured=self.settings.deepseek_configured,
                occurred_at=utcstr(now_utc())))
            return self.get(actor)

    def test(self, actor):
        self._authorize(actor)
        if not self.settings.deepseek_configured:
            return dict(ok=False, message="请先填写API Key并保存配置。", error_code="deepseek_not_configured")
        adapter, started = None, perf_counter()
        try:
            self._validate("deepseek", str(self.settings.deepseek_base_url), self.settings.deepseek_model, None)
            configuration = self.settings.model_copy(update={"deepseek_max_tokens":16,
                "deepseek_max_retries":0,"deepseek_timeout_seconds":min(10,self.settings.deepseek_timeout_seconds)})
            adapter = self.adapter_factory(configuration)
            messages = [{"role":"user", "content":"Reply OK"}]
            record_disclosure(self.store, dict(task_id=str(uuid4()), source_id="synthetic-connection-check", session_id="connection-check"), messages, [])
            response = adapter.chat(messages)
            result = dict(ok=True, message="DeepSeek连接成功。", model=configuration.deepseek_model)
        except DeepSeekRequestError as error:
            code = error.status_code if type(error.status_code) is int else None
            message = ({401:"鉴权失败，请检查API Key。",403:"接口访问被拒绝，请检查账号权限。",
                402:"账号余额不足，请检查DeepSeek账号。",429:"请求频率受限，请稍后重试。"}.get(code)
                or ("DeepSeek服务暂不可用。" if code is not None and code >= 500 else "连接失败，请检查网络和配置。"))
            result = dict(ok=False, message=message, error_code="model_connection_failed")
            if code is not None:
                result["status_code"] = code
        except Exception:
            result = dict(ok=False, message="连接测试未完成，请检查网络和模型配置。", error_code="model_connection_failed")
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    pass
        result["latency_ms"] = round((perf_counter()-started)*1000)
        return result
