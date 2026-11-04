"""经认证的业务HTTP接口。"""

from datetime import datetime
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from app.schemas.common import DomainError

router = APIRouter(prefix="/api")


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class Login(Payload):
    username: str
    password: str


class Source(Payload):
    source_id: str
    name: str
    face_id: str
    timezone: str = "+08:00"


class Point(Payload):
    point_id: str
    source_id: str
    source_tag: str
    name: str
    quantity: str
    unit: str
    unit_status: str = "inferred"
    mapping_status: str = "confirmed"
    lineage: str = "raw"


class Import(Payload):
    source_id: str
    file_id: str
    mapping: dict
    start_at: AwareDatetime | None = None
    end_at: AwareDatetime | None = None


class Session(Payload):
    source_id: str
    mode: str
    start_as_of: AwareDatetime
    config_version: str = "demo-v1"
    speed: float = 1
    parent_session_id: str | None = None


class SessionAction(Payload):
    action: str
    speed: float | None = None
    target_as_of: AwareDatetime | None = None


class Tick(Payload):
    target_as_of: AwareDatetime | None = None


class EventBrief(Payload):
    session_id: str = Field(min_length=1)
    event_revision: int | None = Field(default=None, ge=1)
    retry: bool = False


class EventBriefCancel(Payload):
    session_id: str = Field(min_length=1)
    event_revision: int | None = Field(default=None, ge=1)


class Forecast(Payload):
    session_id: str
    point_id: str
    model_id: str | None = None
    version: str | None = None
    horizon_seconds: int | None = None


class ModelPackageRun(Payload):
    session_id: str
    model_id: str
    version: str | None = None
    features: dict


class Search(Payload):
    session_id: str
    query: str
    event_id: str | None = None


class Advice(Payload):
    session_id: str
    event_id: str
    reference_ids: list[str]


class EventAdvice(Payload):
    session_id: str
    reference_ids: list[str]


class CaseAction(Payload):
    session_id: str
    action_type: str
    operation_context: str
    content: str = ""
    advice_id: str | None = None
    advice_version: int | None = None
    review_decision: str | None = None


class Chat(Payload):
    session_id: str
    message: str | dict
    async_execution: bool = False


class Review(Payload):
    version: str | int
    decision: str
    comment: str = ""


def services(request: Request):
    return request.app.state.services


def token(authorization: Annotated[str | None, Header()] = None):
    if not authorization or not authorization.startswith("Bearer "):
        raise DomainError("请登录", "unauthorized", 401)
    return authorization[7:]


def actor(s=Depends(services), access_token=Depends(token)):
    return s["auth"].get_actor(access_token)


def _object(value, name):
    if not isinstance(value, dict):
        raise DomainError(f"{name}须为JSON对象")
    return value


@router.post("/auth/login")
def login(body: Login, s=Depends(services)):
    return s["auth"].login(body.username, body.password)


@router.get("/auth/me")
def me(a=Depends(actor)):
    return a.model_dump()


@router.post("/auth/logout")
def logout(a=Depends(actor), access_token=Depends(token), s=Depends(services)):
    s["auth"].logout(access_token)
    return {"status": "ok"}


@router.get("/sources")
def sources(a=Depends(actor), s=Depends(services)):
    return s["data"].list_sources(a)


@router.post("/sources")
def register_source(body: Source, a=Depends(actor), s=Depends(services)):
    s["auth"].authorize(a, "manage_data", body.face_id)
    return s["data"].register_source(**body.model_dump())


@router.get("/points")
def points(source_id: str | None = None, a=Depends(actor), s=Depends(services)):
    return s["data"].list_points(a, source_id)


@router.get("/sources/{source_id}/coverage")
def source_coverage(source_id: str, a=Depends(actor), s=Depends(services)):
    return s["data"].get_coverage(a, source_id)


@router.post("/points")
def register_point(body: Point, a=Depends(actor), s=Depends(services)):
    source = s["data"]._source(body.source_id)
    s["auth"].authorize(a, "manage_data", source["face_id"])
    return s["data"].register_point(**body.model_dump())


@router.post("/files")
def upload(request: Request, file: UploadFile = File(), a=Depends(actor), s=Depends(services)):
    s["auth"].authorize(a, "manage_data")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".csv", ".xlsx"):
        raise DomainError("监测文件须为CSV或XLSX")
    staging = request.app.state.settings.runtime_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    path = None
    try:
        with NamedTemporaryFile(dir=staging, suffix=suffix, delete=False) as output:
            path = Path(output.name)
            size = 0
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > 256 * 1024 * 1024:
                    raise DomainError("单文件上传上限256MB")
                output.write(chunk)
        return s["data"].register_file(path, a)
    finally:
        if path:
            path.unlink(missing_ok=True)
        file.file.close()


@router.post("/history-imports")
@router.post("/history/imports", include_in_schema=False)
def history_import(body: Import, a=Depends(actor), s=Depends(services)):
    return s["data"].import_history(a, **body.model_dump())


@router.post("/ingest")
def ingest(body: dict = Body(), a=Depends(actor), s=Depends(services)):
    if set(body) != {"source_id", "readings"}:
        raise DomainError("实时接收需要source_id与readings")
    return s["data"].ingest_readings(a, body["source_id"], body["readings"])


@router.get("/sessions")
def list_sessions(a=Depends(actor), s=Depends(services)):
    return s["sessions"].list_sessions(a)


@router.post("/sessions")
def create_session(body: Session, a=Depends(actor), s=Depends(services)):
    return s["sessions"].create_session(a, **body.model_dump())


@router.get("/sessions/{session_id}")
def get_session(session_id: str, a=Depends(actor), s=Depends(services)):
    return s["sessions"].get_session(a, session_id)


@router.post("/sessions/{session_id}/actions")
def session_action(session_id: str, body: SessionAction, a=Depends(actor), s=Depends(services)):
    return s["sessions"].apply_action(a, session_id, **body.model_dump())


@router.post("/sessions/{session_id}/tick")
def tick(session_id: str, body: Tick, background_tasks: BackgroundTasks, a=Depends(actor), s=Depends(services)):
    return s["runner"].run_once(a, session_id, body.target_as_of,
                                schedule=lambda work: background_tasks.add_task(work))


@router.get("/observations")
def observations(session_id: str, point_id: str, start_at: AwareDatetime, end_at: AwareDatetime,
                 aggregation_seconds: int | None = None, fill_max_seconds: int = 0,
                 limit: int = 1000, offset: int = 0, a=Depends(actor), s=Depends(services)):
    ctx = s["sessions"].get_context(a, session_id)
    return s["data"].query_series(ctx, point_id, start_at, end_at, aggregation_seconds, fill_max_seconds, limit, offset)


@router.get("/models")
def models(session_id: str, point_id: str | None = None, a=Depends(actor), s=Depends(services)):
    return s["prediction"].list_models(s["sessions"].get_context(a, session_id), point_id)


@router.get("/model-packages")
def model_packages(session_id: str, a=Depends(actor), s=Depends(services)):
    return s["model_packages"].list_packages(s["sessions"].get_context(a, session_id))


@router.post("/model-package-runs")
def run_model_package(body: ModelPackageRun, a=Depends(actor), s=Depends(services)):
    return s["model_packages"].run(s["sessions"].get_context(a, body.session_id), body.model_id, body.features, body.version)


@router.get("/model-package-runs/{run_id}")
def get_model_package_run(run_id: str, session_id: str, a=Depends(actor), s=Depends(services)):
    return s["model_packages"].get_run(s["sessions"].get_context(a, session_id), run_id)


@router.post("/models")
def register_model(body: dict = Body(), a=Depends(actor), s=Depends(services)):
    return s["prediction"].register_model(a, body)


@router.post("/forecasts")
def forecast(body: Forecast, a=Depends(actor), s=Depends(services)):
    return s["prediction"].run_prediction(s["sessions"].get_context(a, body.session_id), body.point_id,
                                          body.model_id, body.horizon_seconds, version=body.version)


@router.get("/forecasts")
def list_forecasts(session_id: str, point_id: str | None = None, model_id: str | None = None,
                   version: str | None = None, limit: int = Query(default=20, ge=1, le=100),
                   a=Depends(actor), s=Depends(services)):
    return s["prediction"].list_forecasts(s["sessions"].get_context(a, session_id), point_id, model_id, version, limit)


@router.get("/forecasts/{run_id}")
def get_forecast(run_id: str, session_id: str, a=Depends(actor), s=Depends(services)):
    return s["prediction"].get_forecast(s["sessions"].get_context(a, session_id), run_id)


@router.get("/rules")
def rules(face_id: str | None = None, point_id: str | None = None, a=Depends(actor), s=Depends(services)):
    return s["alerts"].get_rules(a, face_id, point_id)


@router.get("/events")
def events(session_id: str, point_id: str | None = None, a=Depends(actor), s=Depends(services)):
    return s["alerts"].query_events(s["sessions"].get_context(a, session_id), point_id=point_id)


@router.get("/evidence")
def evidence(session_id: str, event_id: str, event_revision: int | None = None, a=Depends(actor), s=Depends(services)):
    return s["traceability"].get_event_evidence(s["sessions"].get_context(a, session_id), event_id, event_revision)


@router.get("/events/{event_id}/brief")
def event_brief(event_id: str, session_id: str, event_revision: int | None = Query(default=None, ge=1),
                a=Depends(actor), s=Depends(services)):
    return s["event_briefs"].get(s["sessions"].get_context(a, session_id), event_id, event_revision)


@router.post("/events/{event_id}/brief")
def submit_event_brief(event_id: str, body: EventBrief, background_tasks: BackgroundTasks,
                       a=Depends(actor), s=Depends(services)):
    return s["event_briefs"].submit(s["sessions"].get_context(a, body.session_id), event_id,
                                   event_revision=body.event_revision, retry=body.retry,
                                   schedule=lambda work: background_tasks.add_task(work))


@router.post("/events/{event_id}/brief/cancel")
def cancel_event_brief(event_id: str, body: EventBriefCancel, a=Depends(actor), s=Depends(services)):
    return s["event_briefs"].cancel(s["sessions"].get_context(a, body.session_id), event_id, body.event_revision)


@router.get("/cases/{event_id}")
@router.get("/events/{event_id}/case", include_in_schema=False)
def get_case(event_id: str, session_id: str, a=Depends(actor), s=Depends(services)):
    return s["cases"].get_case(s["sessions"].get_context(a, session_id), event_id)


@router.post("/cases/{event_id}/actions")
@router.post("/events/{event_id}/case/actions", include_in_schema=False)
def case_action(event_id: str, body: CaseAction, a=Depends(actor), s=Depends(services)):
    return s["cases"].record_action(a, s["sessions"].get_context(a, body.session_id), event_id, body.model_dump(exclude={"session_id"}, exclude_none=True))


@router.get("/references")
def references(state: str | None = None, kind: str | None = None, face_id: str | None = None, a=Depends(actor), s=Depends(services)):
    return s["knowledge"].list_references(a, state, kind, face_id)


@router.post("/references/files")
async def upload_reference_file(file: UploadFile = File(), face_ids: str | None = Form(default=None),
                                a=Depends(actor), s=Depends(services)):
    s["knowledge"].auth.authorize(a, "manage_data")
    faces = None
    if face_ids is not None:
        try:
            faces = json.loads(face_ids)
        except (ValueError, TypeError):
            raise DomainError("工作面范围需要JSON列表") from None
        if not isinstance(faces, list) or not faces or any(not isinstance(face, str) or not face.strip() for face in faces):
            raise DomainError("工作面范围需要非空字符串列表")
    try:
        content = await file.read(20 * 1024 * 1024 + 1)
        return s["knowledge"].upload_file(a, file.filename or "", content, faces)
    finally:
        await file.close()


@router.get("/references/files/{file_id}")
def reference_file(file_id: str, a=Depends(actor), s=Depends(services)):
    return s["knowledge"].get_file(a, file_id)


@router.get("/references/files/{file_id}/content")
def reference_file_content(file_id: str, a=Depends(actor), s=Depends(services)):
    metadata, content = s["knowledge"].get_file_content(a, file_id)
    return Response(content, media_type=metadata["media_type"], headers={
        "Content-Disposition": "attachment; filename*=UTF-8''" + quote(metadata["filename"], safe=""),
        "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
    })


@router.post("/references/imports")
def reference_import(body: dict = Body(), a=Depends(actor), s=Depends(services)):
    return s["knowledge"].import_reference(a, _object(body.get("spec", body), "资料内容"))


@router.post("/references/{entry_id}/review")
def review_reference(entry_id: str, body: Review, a=Depends(actor), s=Depends(services)):
    return s["knowledge"].review_reference(a, entry_id, **body.model_dump())


@router.post("/references/search")
def search(body: Search, a=Depends(actor), s=Depends(services)):
    ctx = s["sessions"].get_context(a, body.session_id)
    scope = s["alerts"].get_event_snapshot(ctx, body.event_id) if body.event_id else None
    return s["knowledge"].search_documents(ctx, body.query, scope)


@router.get("/references/{entry_id}")
def reference(entry_id: str, version: str | None = None, a=Depends(actor), s=Depends(services)):
    return s["knowledge"].get_reference(a, entry_id, version)


@router.post("/advice")
def advice(body: Advice, a=Depends(actor), s=Depends(services)):
    ctx = s["sessions"].get_context(a, body.session_id)
    bundle = s["traceability"].get_event_evidence(ctx, body.event_id)
    return s["knowledge"].draft_advice(ctx, bundle, body.reference_ids)


@router.post("/experiences")
def experience(body: dict = Body(), a=Depends(actor), s=Depends(services)):
    snapshot = s["cases"].export_reviewed_case(a, body.get("case_id"))
    return s["knowledge"].submit_experience(a, snapshot, _object(body.get("spec", {}), "经验内容"))


@router.post("/experiences/{experience_id}/review")
def review_experience(experience_id: str, body: Review, a=Depends(actor), s=Depends(services)):
    return s["knowledge"].review_experience(a, experience_id, **body.model_dump())


@router.post("/chat")
def chat(body: Chat, background_tasks: BackgroundTasks, a=Depends(actor), s=Depends(services)):
    ctx = s["sessions"].get_context(a, body.session_id)
    if body.async_execution:
        return s["agent"].submit(ctx, body.message, schedule=lambda work: background_tasks.add_task(work))
    return s["agent"].chat(ctx, body.message)


@router.get("/tasks/{task_id}")
def task(task_id: str, a=Depends(actor), s=Depends(services)):
    return s["agent"].get_task(a, task_id)


@router.post("/tasks/{task_id}/cancel")
def cancel(task_id: str, a=Depends(actor), s=Depends(services)):
    return s["agent"].cancel_task(a, task_id)


@router.get("/evaluations")
def evaluations(a=Depends(actor), s=Depends(services)):
    s["auth"].authorize(a, "evaluate")
    return [r for r in s["store"].list("evaluation_report") if r["face_id"] in a.face_ids]


@router.post("/evaluations")
def evaluate(body: dict = Body(), a=Depends(actor), s=Depends(services)):
    return s["evaluation"].run(a, _object(body.get("spec", {}), "评估清单"), _object(body.get("records", {}), "冻结记录"))


@router.get("/events/{event_id}/evidence", include_in_schema=False)
def event_evidence(event_id: str, session_id: str, event_revision: int | None = None, a=Depends(actor), s=Depends(services)):
    return s["traceability"].get_event_evidence(s["sessions"].get_context(a, session_id), event_id, event_revision)


@router.post("/events/{event_id}/advice", include_in_schema=False)
def event_advice(event_id: str, body: EventAdvice, a=Depends(actor), s=Depends(services)):
    request = Advice(event_id=event_id, **body.model_dump())
    return advice(request, a, s)


@router.get("/evaluations/{evaluation_id}")
def evaluation(evaluation_id: str, a=Depends(actor), s=Depends(services)):
    return s["evaluation"].get_report(a, evaluation_id)
