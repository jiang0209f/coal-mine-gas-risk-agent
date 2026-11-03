"""本地后端入口。"""

from fastapi import FastAPI

from app.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    configuration = settings if settings is not None else Settings()
    application = FastAPI(title="煤矿瓦斯智能体", version="0.1.0")

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "coal-mine-gas-risk-agent",
            "deepseek": {
                "configured": configuration.deepseek_configured,
                "model": configuration.deepseek_model,
            },
        }

    return application


app = create_app()
