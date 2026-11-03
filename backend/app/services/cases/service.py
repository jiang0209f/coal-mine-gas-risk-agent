"""人工确认、建议审核、执行反馈与复核快照。"""

from datetime import datetime
from uuid import uuid4

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext, now_utc, utcstr
from app.services.auth.service import AuthService
from app.services.knowledge.service import KnowledgeService

__all__ = ["CasesService"]


class CasesService:
    def __init__(self, store: Store, alerts=None, knowledge=None, clock=now_utc) -> None:
        self.store = store
        self.alerts = alerts
        self.knowledge = knowledge or KnowledgeService(store)
        self.auth = AuthService(store)
        self.clock = clock

    def get_case(self, context: ExecutionContext, event_id: str) -> dict:
        event = self._event(context, event_id)
        self.auth.authorize(context.actor, "read", event["face_id"])
        advice = self.knowledge.list_advice(context.actor, event_id)
        with self.store.transaction() as connection:
            case = self._find_case(context.session_id, event_id, connection)
            actions = [self.store.get("case_action", ref, connection)
                       for ref in case["action_refs"]] if case else []
        if case is None:
            case = {"case_id": None, "event_id": event_id, "phase": "new",
                    "operation_context": None, "action_refs": [], "advice_review_status": {}}
        pending = {
            "new": ["confirm_event"], "confirmed": ["review_advice"],
            "advice_reviewed": ["record_execution"], "executed": ["submit_feedback"],
            "feedback_submitted": ["review_case"], "reviewed": [],
        }[case["phase"]]
        if case["phase"] == "confirmed" and not advice:
            pending = ["draft_advice"]
        if case["phase"] == "advice_reviewed":
            decisions = case["advice_review_status"]
            approved = any(status["decision"] == "approved" for status in decisions.values())
            unreviewed = any(f"{entry['advice_id']}:{entry['version']}" not in decisions for entry in advice)
            pending = (["review_advice"] if unreviewed else []) + (["record_execution"] if approved else ["draft_advice"])
        return {"case": case, "actions": actions, "advice_versions": advice,
                "advice_review_status": case["advice_review_status"], "pending_items": pending}

    def record_action(
        self, actor: Actor, context: ExecutionContext, event_id: str, action: dict,
    ) -> dict:
        if actor != context.actor:
            raise DomainError("操作身份与会话身份不一致", code="forbidden", status_code=403)
        action_type = action.get("action_type")
        self.auth.authorize(actor, action_type)
        event = self._event(context, event_id)
        actor.require_face(event["face_id"])
        operation_context = action.get("operation_context")
        if operation_context not in ("rehearsal", "field_record"):
            raise DomainError("操作必须标记field_record或rehearsal")
        if context.mode != "realtime" and operation_context != "rehearsal":
            raise DomainError("历史与模拟会话的操作必须标为演练", code="forbidden", status_code=403)
        content = action.get("content", "")
        if not isinstance(content, str) or (action_type != "review_advice" and not content.strip()):
            raise DomainError("该操作需要非空内容")
        with self.store.transaction() as connection:
            case = self._find_case(context.session_id, event_id, connection)
            if case is None:
                if action_type != "confirm_event":
                    raise self._flow_error("请先人工确认事件")
                case = {
                    "case_id": str(uuid4()), "event_id": event_id,
                    "session_id": context.session_id, "face_id": event["face_id"],
                    "phase": "new", "operation_context": operation_context,
                    "action_refs": [], "advice_review_status": {},
                }
            if case["phase"] == "reviewed":
                raise self._flow_error("已复核的处置记录已经冻结")
            if case["operation_context"] != operation_context:
                raise self._flow_error("同一处置单的来源性质必须一致")
            advice = None
            if action_type == "confirm_event":
                if case["phase"] != "new":
                    raise self._flow_error("事件已经确认")
                case["phase"] = "confirmed"
            elif action_type in ("review_advice", "record_execution"):
                if case["phase"] not in ("confirmed", "advice_reviewed"):
                    raise self._flow_error("当前阶段不能审核或执行建议")
                advice = self._advice(actor, context, event_id, action, connection)
                if operation_context == "field_record" and advice["operation_context"] == "rehearsal":
                    raise DomainError("现场处置不能采用演练建议", code="forbidden", status_code=403)
                advice_key = f"{advice['advice_id']}:{advice['version']}"
                if action_type == "review_advice":
                    if action.get("review_decision") not in ("approved", "rejected"):
                        raise DomainError("建议审核结论必须为approved或rejected")
                    case["advice_review_status"][advice_key] = {
                        "decision": action["review_decision"], "action_ref": None,
                    }
                    case["phase"] = "advice_reviewed"
                else:
                    status = case["advice_review_status"].get(advice_key, {})
                    if status.get("decision") != "approved":
                        raise self._flow_error("执行必须关联已批准的具体建议版本")
                    case["phase"] = "executed"
            elif action_type == "submit_feedback":
                if case["phase"] != "executed":
                    raise self._flow_error("请先记录建议执行")
                case["phase"] = "feedback_submitted"
            elif action_type == "review_case":
                if case["phase"] != "feedback_submitted":
                    raise self._flow_error("请先提交执行反馈")
                case["phase"] = "reviewed"
            else:
                raise DomainError("处置动作无效")
            action_id = str(uuid4())
            record = {
                "action_id": action_id, "case_id": case["case_id"], "event_id": event_id,
                "action_type": action_type, "actor_id": actor.user_id,
                "role": "safety_management" if action_type == "review_advice" else actor.roles[0],
                "occurred_at": utcstr(self.clock()), "data_as_of": utcstr(context.as_of),
                "content": content, "operation_context": operation_context,
                "advice_id": advice["advice_id"] if advice else None,
                "advice_version": advice["version"] if advice else None,
                "review_decision": action.get("review_decision") if action_type == "review_advice" else None,
            }
            case["action_refs"].append(action_id)
            if action_type == "review_advice":
                case["advice_review_status"][advice_key]["action_ref"] = action_id
            self.store.put("case_action", action_id, record, connection)
            self.store.put("case", case["case_id"], case, connection)
            if action_type == "review_case":
                self._freeze_snapshot(case, record, connection)
        return {"case": case, "case_action": record}

    def export_reviewed_case(self, actor: Actor, case_id: str) -> dict:
        case = self.store.get("case", case_id)
        if case is None:
            raise DomainError("处置单不存在", code="not_found", status_code=404)
        self.auth.authorize(actor, "read", case["face_id"])
        if case["phase"] != "reviewed":
            raise self._flow_error("处置单尚未完成复核")
        snapshot = self.store.get("reviewed_case_snapshot", case_id)
        if snapshot is None:
            raise self._flow_error("处置单缺少已复核快照")
        return snapshot

    def _event(self, context, event_id) -> dict:
        event = (self.alerts.get_event_snapshot(context, event_id) if self.alerts is not None
                 else self.store.get("event", event_id))
        if event is None:
            raise DomainError("事件不存在", code="not_found", status_code=404)
        if event.get("session_id") != context.session_id:
            raise DomainError("事件不属于当前会话", code="forbidden", status_code=403)
        if event.get("source_id") != context.source_id:
            raise DomainError("事件不属于当前数据源", code="forbidden", status_code=403)
        try:
            revision_as_of = datetime.fromisoformat(event["revision_as_of"])
            utcstr(revision_as_of)
        except (KeyError, ValueError, TypeError) as exc:
            raise DomainError("事件缺少有效的修订时间") from exc
        if revision_as_of > context.as_of:
            raise self._flow_error("事件修订在当前数据时间尚不可见")
        point = self.store.get("point", event.get("point_id", ""))
        if point is not None and point.get("source_id") != context.source_id:
            raise DomainError("事件测点不属于当前数据源", code="forbidden", status_code=403)
        source = self.store.get("source", context.source_id)
        face_id = event.get("face_id") or (point or {}).get("face_id") or (source or {}).get("face_id")
        if not face_id:
            raise DomainError("事件缺少工作面映射")
        context.actor.require_face(face_id)
        if source is None or source["face_id"] != face_id:
            raise DomainError("事件工作面与会话来源不一致", code="forbidden", status_code=403)
        return {**event, "face_id": face_id}

    def _find_case(self, session_id, event_id, connection):
        return next((case for case in self.store.list("case", connection)
                     if case["session_id"] == session_id and case["event_id"] == event_id), None)

    def _advice(self, actor, context, event_id, action, connection) -> dict:
        advice_id, version = action.get("advice_id"), action.get("advice_version")
        if not advice_id or type(version) is not int or version < 1:
            raise DomainError("建议操作需要advice_id和advice_version")
        advice = self.store.get("advice", f"{advice_id}:{version}", connection)
        if advice is None:
            raise DomainError("建议版本不存在", code="not_found", status_code=404)
        actor.require_face(advice["face_id"])
        if advice["event_id"] != event_id or advice["session_id"] != context.session_id:
            raise DomainError("建议不属于当前事件", code="forbidden", status_code=403)
        return advice

    def _freeze_snapshot(self, case, review, connection) -> None:
        actions = [self.store.get("case_action", ref, connection) for ref in case["action_refs"]]
        executions = [action for action in actions if action["action_type"] == "record_execution"]
        advice_refs = [self.store.get("advice", f"{action['advice_id']}:{action['advice_version']}", connection)
                       for action in executions]
        snapshot = {
            "snapshot_ref": str(uuid4()), "case_id": case["case_id"],
            "event_id": case["event_id"], "face_id": case["face_id"],
            "review_action_ref": review["action_id"], "operation_context": case["operation_context"],
            "advice_refs": advice_refs,
            "feedback_refs": [action["action_id"] for action in actions if action["action_type"] == "submit_feedback"],
            "reviewed_at": review["occurred_at"], "actions": actions,
            "source_refs": list(dict.fromkeys(ref for advice in advice_refs for ref in advice["reference_ids"])),
        }
        self.store.put("reviewed_case_snapshot", case["case_id"], snapshot, connection)

    @staticmethod
    def _flow_error(message):
        return DomainError(message, code="conflict", status_code=409)
