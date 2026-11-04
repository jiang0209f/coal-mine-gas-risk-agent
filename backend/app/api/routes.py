"""经认证的业务HTTP接口。"""

from datetime import datetime
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal
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


class SessionPredictionModel(Payload):
    point_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class EventBrief(Payload):
    session_id: str = Field(min_length=1)
    event_revision: int | None = Field(default=None, ge=1)
    retry: bool = False


class EventBriefCancel(Payload):
    session_id: str = Field(min_length=1)
    event_revision: int | None = Field(default=None, ge=1)


class SessionEvaluation(Payload):
    session_id: str = Field(min_length=1)
    start_at: AwareDatetime
    end_at: AwareDatetime
    point_ids: list[str] | None = Field(default=None, min_length=1)
    model_id: str | None = None
    model_version: str | None = None
    prediction_stage: str | None = None
    dataset_split: Literal["unclassified", "train", "validation", "test"] = "unclassified"
    tolerance_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)


class IngestionConnector(Payload):
    connector_id: str | None = Field(default=None, min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    source_id: str = Field(min_length=1)
    adapter_kind: Literal["standard_json", "mapped_json"] = "standard_json"
    mapping: dict = Field(default_factory=dict)
    tag_map: dict[str, str] = Field(default_factory=dict)
    enabled: bool = False
    stale_after_seconds: int = Field(default=120, ge=1)


class IngestionAction(Payload):
    action: Literal["enable", "pause", "retry"]


class IngestionBatch(Payload):
    batch_id: str = Field(min_length=1, max_length=200)
    payload: dict | list


class IngestionHeartbeat(Payload):
    pass


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
    scope: Literal["local", "public"] = "local"


class PublicKnowledgeQuestion(Payload):
    session_id: str
    query: str = Field(min_length=1, max_length=20000)


class KnowledgeDisclosure(Payload):
    version: str | int
    disclosure: Literal["local_only", "public_normative"]
    official_url: str | None = None
    category: Literal["law", "regulation", "standard"] | None = None
    comment: str = Field(min_length=1, max_length=2000)


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
    point_id: str | None = Field(default=None, min_length=1)
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


@router.get("/ingestion/status")
def ingestion_status(request: Request, source_id: str | None = None, a=Depends(actor), s=Depends(services)):
    connectors = s["ingestion"].list_connectors(a, source_id)
    status = request.app.state.ingestion_worker_status
    return {"source_id": source_id, "connectors": connectors,
            "worker": {key: status.get(key) for key in ("enabled", "state", "last_run_at", "last_success_at", "last_error_code")}}


@router.get("/ingestion/connectors")
def ingestion_connectors(source_id: str | None = None, a=Depends(actor), s=Depends(services)):
    return s["ingestion"].list_connectors(a, source_id)


@router.post("/ingestion/connectors")
def create_ingestion_connector(body: IngestionConnector, a=Depends(actor), s=Depends(services)):
    return s["ingestion"].create_connector(a, body.model_dump(exclude_none=True))


@router.get("/ingestion/connectors/{connector_id}")
def ingestion_connector(connector_id: str, a=Depends(actor), s=Depends(services)):
    return s["ingestion"].get_connector(a, connector_id)


@router.post("/ingestion/connectors/{connector_id}/actions")
def ingestion_action(connector_id: str, body: IngestionAction, a=Depends(actor), s=Depends(services)):
    return s["ingestion"].action(a, connector_id, body.action)


@router.post("/ingestion/connectors/{connector_id}/heartbeat")
def ingestion_heartbeat(connector_id: str, body: IngestionHeartbeat | None = None, a=Depends(actor), s=Depends(services)):
    return s["ingestion"].heartbeat(a, connector_id)


@router.post("/ingestion/connectors/{connector_id}/batches")
def submit_ingestion_batch(connector_id: str, body: IngestionBatch, a=Depends(actor), s=Depends(services)):
    return s["ingestion"].submit_batch(a, connector_id, body.batch_id, body.payload)


@router.get("/ingestion/connectors/{connector_id}/batches")
def ingestion_batches(connector_id: str, offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=200),
                      a=Depends(actor), s=Depends(services)):
    return s["ingestion"].list_batches(a, connector_id, offset, limit)


@router.get("/ingestion/connectors/{connector_id}/batches/{batch_id:path}")
def ingestion_batch(connector_id: str, batch_id: str, a=Depends(actor), s=Depends(services)):
    return s["ingestion"].get_batch(a, connector_id, batch_id)


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


@router.post("/sessions/{session_id}/prediction-models")
def session_prediction_model(session_id: str, body: SessionPredictionModel, a=Depends(actor), s=Depends(services)):
    return s["sessions"].set_prediction_model(a, session_id, **body.model_dump())


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
def forecast(body: Forecast, background_tasks: BackgroundTasks, a=Depends(actor), s=Depends(services)):
    return s["runner"].run_forecast(s["sessions"].get_context(a, body.session_id), body.point_id,
        body.model_id, body.horizon_seconds, version=body.version,
        schedule=lambda work: background_tasks.add_task(work))


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
    if body.scope == "public":
        return s["knowledge"].search_public_documents(ctx, body.query)
    return s["knowledge"].search_documents(ctx, body.query, scope)


@router.post("/references/public-answer")
def public_knowledge_answer(body: PublicKnowledgeQuestion, a=Depends(actor), s=Depends(services)):
    context = s["sessions"].get_context(a, body.session_id)
    return s["public_knowledge"].run(context, body.query, public_query_confirmed=True)


@router.post("/references/{entry_id}/disclosure")
def knowledge_disclosure(entry_id: str, body: KnowledgeDisclosure, a=Depends(actor), s=Depends(services)):
    return s["knowledge"].set_disclosure(a, entry_id, **body.model_dump())


@router.get("/references/fragments/{reference_id}")
def reference_fragment(reference_id: str, a=Depends(actor), s=Depends(services)):
    fragment = s["store"].get("reference_fragment", reference_id)
    if fragment is None:
        raise DomainError("引用不存在", code="not_found", status_code=404)
    document = s["knowledge"].get_reference(a, fragment["entry_id"], fragment["version"])
    return {**fragment, "title": document.get("title"), "state": document["state"]}


@router.post("/references/conflicts")
def register_knowledge_conflict(body: dict = Body(...), a=Depends(actor), s=Depends(services)):
    return s["knowledge"].set_conflict(a, body)


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
    message = body.message
    if body.point_id is not None:
        point = s["data"]._point(body.point_id)
        if point["source_id"] != ctx.source_id:
            raise DomainError("问答测点不属于当前来源", code="forbidden", status_code=403)
        if isinstance(message, dict) and message.get("selected_point_id") not in (None, body.point_id):
            raise DomainError("问答测点与当前选择不一致")
        message = ({**message, "selected_point_id": body.point_id} if isinstance(message, dict)
                   else {"message": message, "selected_point_id": body.point_id})
    if body.async_execution:
        return s["agent"].submit(ctx, message, schedule=lambda work: background_tasks.add_task(work))
    return s["agent"].chat(ctx, message)


@router.get("/tasks/{task_id}")
def task(task_id: str, a=Depends(actor), s=Depends(services)):
    return s["agent"].get_task(a, task_id)


@router.post("/tasks/{task_id}/cancel")
def cancel(task_id: str, a=Depends(actor), s=Depends(services)):
    return s["agent"].cancel_task(a, task_id)


@router.get("/evaluations")
def evaluations(a=Depends(actor), s=Depends(services)):
    s["auth"].authorize(a, "evaluate")
    return [s["evaluation"].get_report(a, r["evaluation_id"])
            for r in s["store"].list("evaluation_report") if r["face_id"] in a.face_ids]


@router.post("/evaluations")
def evaluate(body: dict = Body(), a=Depends(actor), s=Depends(services)):
    return s["evaluation"].run(a, _object(body.get("spec", {}), "评估清单"), _object(body.get("records", {}), "冻结记录"))


@router.post("/evaluations/from-session")
def evaluate_session(body: SessionEvaluation, a=Depends(actor), s=Depends(services)):
    s["auth"].authorize(a, "evaluate")
    context = s["sessions"].get_context(a, body.session_id)
    collected = s["evaluation_collector"].collect(context, **body.model_dump(exclude={"session_id"}))
    manifest = s["evaluation"].freeze_manifest(a, collected["spec"],
        {**collected.get("artifact_refs", {}), "frozen_records": collected["records"]})
    return s["evaluation"].build_report(manifest["manifest_id"])


@router.get("/evaluations/{evaluation_id}/manifest")
def evaluation_manifest(evaluation_id: str, a=Depends(actor), s=Depends(services)):
    return s["evaluation"].get_manifest(a, evaluation_id)


@router.get("/evaluations/{evaluation_id}/metrics/{metric}")
def evaluation_metric(evaluation_id: str, metric: str, variant: str = "primary",
                      offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=200),
                      a=Depends(actor), s=Depends(services)):
    return s["evaluation"].get_metric_details(a, evaluation_id, metric, variant, offset, limit)


@router.get("/evaluations/{evaluation_id}/export")
def export_evaluation(evaluation_id: str, a=Depends(actor), s=Depends(services)):
    exported = s["evaluation"].export_report(a, evaluation_id)
    return Response(json.dumps(exported, ensure_ascii=False, allow_nan=False), media_type="application/json", headers={
        "Content-Disposition": f'attachment; filename="evaluation-{evaluation_id}.json"', "Cache-Control": "no-store",
    })


@router.get("/evaluations/{evaluation_id}/records/observations/{observation_id}")
def evaluation_observation(evaluation_id: str, observation_id: str, a=Depends(actor), s=Depends(services)):
    return s["evaluation"].get_frozen_observation(a, evaluation_id, observation_id)


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
