"""本地后端入口。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT, Settings
from app.db import Store
from app.schemas.common import DomainError


def build_services(configuration):
    from app.services.auth.service import AuthService
    from app.services.data.service import DataService
    from app.services.sessions.service import SessionsService
    from app.services.sessions.runner import SessionRunner
    from app.services.prediction.service import PredictionService
    from app.services.prediction.packages import ModelPackagesService
    from app.services.prediction.trained import TrainedModelsPipeline
    from app.adapters.model_runtime import ModelRuntime
    from app.services.alerts.service import AlertsService
    from app.services.traceability.service import TraceabilityService
    from app.services.knowledge.service import KnowledgeService
    from app.services.cases.service import CasesService
    from app.services.evaluation.service import EvaluationService
    from app.services.agent.service import AgentService
    from app.services.agent.briefs import EventBriefsService

    runtime = configuration.runtime_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    store = Store(configuration.database_url or f"sqlite+pysqlite:///{(runtime / 'app.db').as_posix()}")
    s = {"store": store, "auth": AuthService(store), "sessions": SessionsService(store)}
    s["data"] = DataService(store, runtime / "imports")
    model_runtime = ModelRuntime(configuration.model_catalog_path)
    trained_models = TrainedModelsPipeline(store, s["data"], model_runtime)
    s["prediction"] = PredictionService(store, s["data"], trained_models)
    s["model_packages"] = ModelPackagesService(store, model_runtime)
    s["alerts"] = AlertsService(store, s["data"])
    s["traceability"] = TraceabilityService(store, s["alerts"], s["data"], s["prediction"], trained_models)
    s["knowledge"] = KnowledgeService(store, upload_dir=runtime / "knowledge-files")
    s["cases"] = CasesService(store, s["alerts"], s["knowledge"])
    s["evaluation"] = EvaluationService(store)
    s["agent"] = AgentService(store, s, configuration)
    s["event_briefs"] = EventBriefsService(store, s, configuration)
    s["event_briefs"].recover_interrupted()
    s["runner"] = SessionRunner(s)
    return s


def create_app(settings: Settings | None = None) -> FastAPI:
    configuration = settings if settings is not None else Settings()
    @asynccontextmanager
    async def lifespan(application):
        application.state.settings = configuration
        application.state.services = build_services(configuration)
        try:
            yield
        finally:
            application.state.services["store"].close()

    application = FastAPI(title="煤矿瓦斯智能体", version="0.2.0", lifespan=lifespan)

    @application.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError):
        return JSONResponse({"code": exc.code, "message": exc.message, "missing_fields": exc.missing_fields}, status_code=exc.status_code)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse({"code": "validation_error", "message": "请求字段格式错误",
                             "errors": [{"loc": e["loc"], "type": e["type"]} for e in exc.errors()]}, status_code=422)

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

    from app.api.routes import router
    application.include_router(router)
    frontend = PROJECT_ROOT / "frontend" / "dist"
    if frontend.is_dir():
        application.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return application


app = create_app()
