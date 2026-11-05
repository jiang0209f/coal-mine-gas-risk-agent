"""Event investigation and control workflow. Reviews are explicit authenticated actions."""
from typing import Literal
from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import Response
from pydantic import Field
from app.api.routes import Payload, actor, services

router=APIRouter(prefix="/api")

class EventRequest(Payload):
    session_id: str
    retry: bool=False

class ReviewRequest(Payload):
    session_id: str
    decision: Literal["approve","reject","approved","rejected"]
    note: str=Field(min_length=1)
    tier: int|None=Field(default=None,ge=1,le=5)
    assigned_role: Literal["ventilation","gas_governance","safety_management"]|None=None
    task_due_at: dict[str,str]|None=None

class MeasuresRequest(Payload):
    session_id: str
    reference_ids: list[str]=Field(min_length=1)

class PlanRequest(Payload):
    session_id: str
    report_id: str

class ExecutionRequest(Payload):
    session_id: str
    content: str=Field(min_length=1)

class CompletionRequest(Payload):
    session_id: str
    note: str=Field(min_length=1)

def context(s,a,sid): return s["sessions"].get_context(a,sid)
def decision(value): return "approved" if value in ("approve","approved") else "rejected"

@router.get("/events/{event_id}/investigation")
def investigation(event_id:str,session_id:str,s=Depends(services),a=Depends(actor)):
    return s["investigation"].for_event(context(s,a,session_id),event_id)

@router.post("/events/{event_id}/investigation")
def start_investigation(event_id:str,body:EventRequest,background:BackgroundTasks,s=Depends(services),a=Depends(actor)):
    return s["investigation"].submit(context(s,a,body.session_id),event_id,
        schedule=background.add_task,retry=body.retry)

@router.get("/investigations/{report_id}")
def get_report(report_id:str,session_id:str,s=Depends(services),a=Depends(actor)):
    return s["investigation"].get(context(s,a,session_id),report_id)

@router.post("/investigations/{report_id}/review")
def review_report(report_id:str,body:ReviewRequest,s=Depends(services),a=Depends(actor)):
    return s["investigation"].review(context(s,a,body.session_id),report_id,decision(body.decision),body.note)

@router.get("/investigations/{report_id}/report.docx")
def export_report(report_id:str,session_id:str,presentation:bool=False,s=Depends(services),a=Depends(actor)):
    content=s["investigation"].export_docx(context(s,a,session_id),report_id,presentation=presentation)
    return Response(content,media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition":f'attachment; filename="investigation-{report_id}.docx"'})

@router.get("/measures")
def measures(session_id:str,s=Depends(services),a=Depends(actor)):
    return s["control"].list_measures(context(s,a,session_id))

@router.post("/measures")
def import_measures(body:MeasuresRequest,s=Depends(services),a=Depends(actor)):
    return s["control"].import_measures(context(s,a,body.session_id),body.reference_ids)

@router.post("/measures/{measure_id}/review")
def review_measure(measure_id:str,body:ReviewRequest,s=Depends(services),a=Depends(actor)):
    return s["control"].review_measure(context(s,a,body.session_id),measure_id,decision(body.decision),body.note,
        tier=body.tier,assigned_role=body.assigned_role)

@router.get("/events/{event_id}/control")
def event_control(event_id:str,session_id:str,s=Depends(services),a=Depends(actor)):
    ctx=context(s,a,session_id)
    s["alerts"].get_event_snapshot(ctx,event_id)
    plans=[row for row in s["control"].list_plans(ctx) if row["event_id"]==event_id]
    workflows=[s["control"].get_workflow(ctx,row["plan_id"]) for row in plans]
    return dict(measures=s["control"].list_measures(ctx),plans=plans,
        tasks=[task for workflow in workflows for task in workflow["tasks"]],
        experiences=[workflow["experience"] for workflow in workflows if workflow.get("experience")],
        effect=workflows[-1].get("effect") if workflows else None)

@router.post("/events/{event_id}/control-plans")
def draft_plan(event_id:str,body:PlanRequest,s=Depends(services),a=Depends(actor)):
    return s["control"].draft_control_plan(context(s,a,body.session_id),event_id,body.report_id)

@router.get("/control-plans/{plan_id}")
def get_plan(plan_id:str,session_id:str,s=Depends(services),a=Depends(actor)):
    return s["control"].get_workflow(context(s,a,session_id),plan_id)

@router.post("/control-plans/{plan_id}/review")
def review_plan(plan_id:str,body:ReviewRequest,s=Depends(services),a=Depends(actor)):
    return s["control"].review_plan(context(s,a,body.session_id),plan_id,decision(body.decision),body.note,task_due_at=body.task_due_at)

@router.post("/control-tasks/{task_id}/execute")
def execute_task(task_id:str,body:ExecutionRequest,s=Depends(services),a=Depends(actor)):
    return s["control"].record_execution(context(s,a,body.session_id),task_id,body.content)

@router.post("/control-plans/{plan_id}/verify")
def verify(plan_id:str,body:EventRequest,s=Depends(services),a=Depends(actor)):
    return s["control"].verify_control_effect(context(s,a,body.session_id),plan_id)

@router.post("/control-plans/{plan_id}/review-completion")
def complete(plan_id:str,body:CompletionRequest,s=Depends(services),a=Depends(actor)):
    return s["control"].review_completion(context(s,a,body.session_id),plan_id,body.note)

@router.post("/control-experiences/{experience_id}/review")
def review_experience(experience_id:str,body:ReviewRequest,s=Depends(services),a=Depends(actor)):
    return s["control"].review_experience(context(s,a,body.session_id),experience_id,decision(body.decision),body.note)
