"""Human-reviewed measures and plans with local execution/effect records."""
from copy import deepcopy
from datetime import datetime, timedelta
from math import isfinite
import re
from uuid import uuid4

from app.schemas.common import Actor, DomainError, now_utc, utcstr

__all__ = ["ControlService"]

_CAUSES = {
    "local_gas_accumulation": ("瓦斯异常积聚", "瓦斯积聚", "局部瓦斯"),
    "ventilation_abnormal": ("通风系统异常", "通风异常", "风量不足", "通风"),
    "gas_extraction_inadequate": ("抽采不足", "抽采"),
    "goaf_gas_emission": ("采空区瓦斯涌出", "采空区"),
    "adjacent_seam_gas": ("邻近层瓦斯涌出", "邻近层"),
    "coal_cutting": ("割煤速度快", "割煤", "采煤机速度"),
}
_ROLES = ("ventilation", "gas_governance", "safety_management")
_BAD_QUALITY = {"invalid", "invalid_value", "non_numeric", "suspected_calibration", "confirmed_calibration", "planned_calibration", "missing", "missing_value", "forward_filled"}


def _moment(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    utcstr(value)
    return value


def _categories(value):
    text = str(value)
    return {key for key, aliases in _CAUSES.items() if key in text or any(alias in text for alias in aliases)}


class ControlService:
    def __init__(self, store, services):
        self.store, self.services = store, services

    def _scope(self, context, review=False):
        context.actor.require_role(*(("safety_management",) if review else _ROLES))
        if review and context.actor.is_system:
            raise DomainError("需要人工审核", "forbidden", 403)
        source = self.store.get("source", context.source_id)
        if source is None:
            raise DomainError("数据源不存在", "not_found", 404)
        context.actor.require_face(source["face_id"])
        return source["face_id"]

    def _get(self, context, kind, identifier, library=False):
        face = self._scope(context)
        row = self.store.get(kind, identifier)
        if row is None:
            raise DomainError("记录不存在", "not_found", 404)
        if row.get("face_id") != face or (not library and (row.get("source_id") != context.source_id or row.get("session_id") != context.session_id)):
            raise DomainError("记录不属于当前范围", "forbidden", 403)
        if not library and row.get("as_of") and _moment(row["as_of"]) > context.as_of:
            raise DomainError("记录在当前数据时刻尚不可见", "forbidden", 403)
        return row

    def _audit(self, context, action, note=""):
        if not isinstance(note, str):
            raise DomainError("说明须为文本")
        return {"action": action, "actor_id": context.actor.user_id, "roles": list(context.actor.roles),
                "recorded_at": utcstr(now_utc()), "data_as_of": utcstr(context.as_of), "note": note}

    def _reference(self, context, reference_id):
        face = self._scope(context)
        return self.services["knowledge"]._approved_fragment(reference_id, face, context.mode == "realtime")

    def _import_reference(self, context, reference_id):
        try:
            return self._reference(context, reference_id), "approved"
        except DomainError:
            fragment = self.store.get("reference_fragment", reference_id)
            if not fragment:
                raise
            entry = self.store.get("reference", f"{fragment['entry_id']}:{fragment['version']}")
            face = self._scope(context)
            if (not entry or entry.get("state") != "pending_review"
                    or face not in entry.get("applicability", {}).get("face_ids", [])
                    or face not in fragment.get("applicability", {}).get("face_ids", [])):
                raise
            return fragment, "pending_review"

    def list_measures(self, context):
        face = self._scope(context)
        return [row for row in self.store.list("control_measure") if row.get("face_id") == face]

    def import_measures(self, context, reference_ids):
        face = self._scope(context, review=True)
        if (not isinstance(reference_ids, list) or not reference_ids or len(reference_ids) > 100
                or any(not isinstance(value, str) or not value.strip() for value in reference_ids)):
            raise DomainError("需要1至100个资料引用")
        output = []
        for reference_id in dict.fromkeys(reference_ids):
            fragment, reference_state = self._import_reference(context, reference_id)
            text = fragment.get("text", "").strip()
            if not text:
                continue
            previous = next((row for row in self.list_measures(context) if row["reference_id"] == reference_id), None)
            if previous:
                output.append(previous)
                continue
            causes = sorted(_categories(text))
            role = ("ventilation" if re.search(r"通风(?:部|科|队|部门|人员)", text)
                    else "gas_governance" if re.search(r"(?:瓦斯治理|抽采)(?:部|科|队|部门|人员)", text)
                    else "safety_management" if re.search(r"安全(?:管理|监察|检查)(?:部|科|队|部门|人员)|安全员", text)
                    else None)
            deadline = re.search(r"(?:在|于)(\d{1,4})(分钟|小时|日|天)内", text)
            seconds = int(deadline[1]) * {"分钟": 60, "小时": 3600, "日": 86400, "天": 86400}[deadline[2]] if deadline else None
            row = {"measure_id": str(uuid4()), "face_id": face, "reference_id": reference_id,
                   "entry_id": fragment.get("entry_id"), "version": fragment.get("version"), "locator": deepcopy(fragment.get("locator", {})),
                   "text": text, "cause_ids": causes, "generic": not causes and any(word in text for word in ("核实", "核查", "检查", "确认", "复核")),
                   "assigned_role": role, "role": role or "待矿方确定", "reference_state": reference_state,
                   "deadline": deadline[0] if deadline else "待矿方确定", "deadline_seconds": seconds,
                   "tier": 1, "applicability": deepcopy(fragment.get("applicability", {"face_ids": [face]})),
                   "state": "pending", "history": [self._audit(context, "import_measure")], "created_at": utcstr(now_utc())}
            self.store.put("control_measure", row["measure_id"], row)
            output.append(row)
        return output

    def review_measure(self, context, measure_id, decision, note="", tier=None, assigned_role=None):
        self._scope(context, review=True)
        row = self._get(context, "control_measure", measure_id, library=True)
        if decision not in ("approved", "rejected"):
            raise DomainError("审核结论须为approved或rejected")
        if tier is not None:
            if type(tier) is not int or not 1 <= tier <= 5:
                raise DomainError("措施层级须为1至5的整数")
            row.update(tier=tier, tier_basis="reviewer")
        if assigned_role is not None:
            if assigned_role not in _ROLES:
                raise DomainError("责任角色须为通风、瓦斯治理或安全管理角色")
            row.update(assigned_role=assigned_role, role=assigned_role, role_basis="reviewer")
        self._reference(context, row["reference_id"])
        row.update(state=decision, reviewed_at=utcstr(now_utc()), reviewed_by=context.actor.user_id)
        row["history"].append(self._audit(context, "review_measure", note))
        self.store.put("control_measure", measure_id, row)
        return row

    def _event(self, context, event_id, event_revision=None):
        face = self._scope(context)
        arguments = {"context": context, "event_id": event_id}
        if event_revision is not None:
            arguments["event_revision"] = event_revision
        event = self.services["alerts"].get_event_snapshot(**arguments)
        if (event.get("session_id") != context.session_id or event.get("source_id") != context.source_id
                or event.get("face_id", face) != face):
            raise DomainError("事件不属于当前范围", "forbidden", 403)
        return event

    def _plan_event(self, context, plan):
        latest = self._event(context, plan["event_id"])
        frozen = self._event(context, plan["event_id"], plan["event_revision"])
        levels = {"blue": 0, "yellow": 1, "orange": 2, "red": 3}
        if levels.get(latest.get("alert_level"), -1) > levels.get(frozen.get("alert_level"), -1):
            raise DomainError("事件风险等级已升级，请重新排查并审核管控计划", "event_risk_upgraded", 409)
        return frozen

    def _report(self, context, event, report_id):
        report = self._get(context, "investigation_report", report_id)
        if report.get("event_id") != event["event_id"] or report.get("event_revision") != event["event_revision"]:
            raise DomainError("溯源报告与当前事件修订不符", "conflict", 409)
        if report.get("state") != "approved":
            raise DomainError("请先人工审核溯源报告", "report_not_approved", 409)
        return report

    @staticmethod
    def _supported(report):
        causes, labels = set(), []
        for row in report.get("supported_causes", []):
            if isinstance(row, str):
                labels.append(row)
                causes.update(_categories(row))
            elif isinstance(row, dict) and row.get("support_status") == "supported":
                label = row.get("cause", row.get("cause_id", ""))
                labels.append(label)
                causes.update(_categories(str(row.get("cause_id", "")) + " " + label))
        return causes, labels

    def _applicable(self, context, measure, report):
        current = self.store.get("control_measure", measure["measure_id"])
        if measure["state"] != "approved" or not current or current["state"] != "approved":
            return False
        try:
            self._reference(context, measure["reference_id"])
        except DomainError:
            return False
        conditions = measure.get("applicability", {}).get("conditions", {})
        actual = report.get("current_conditions", {})
        return not conditions or (isinstance(conditions, dict) and isinstance(actual, dict)
                                  and all(actual.get(key) == value for key, value in conditions.items()))

    def draft_control_plan(self, context, event_id, report_id):
        event = self._event(context, event_id)
        report = self._report(context, event, report_id)
        causes, labels = self._supported(report)
        measures = [row for row in self.list_measures(context) if self._applicable(context, row, report)
                    and (bool(causes.intersection(row["cause_ids"])) or (not causes and row["generic"]))]
        experiences = [row for row in self.store.list("control_experience") if row.get("face_id") == self._scope(context)
                       and row.get("state") == "approved" and causes.intersection(row.get("cause_ids", []))
                       and _moment(row["effect"]["as_of"]) <= context.as_of
                       and not (context.mode == "realtime" and row.get("operation_context") == "rehearsal")]
        preferred = {identifier for row in experiences for identifier in row.get("measure_ids", [])}
        selected = set()
        for cause in causes or {None}:
            pool = [row for row in measures if cause is None or cause in row["cause_ids"]]
            known = [row for row in pool if row["measure_id"] in preferred]
            level = min((row["tier"] for row in known or pool), default=1)
            selected.update(row["measure_id"] for row in pool if row["tier"] == level)
        measures = [row for row in measures if row["measure_id"] in selected]
        measures.sort(key=lambda row: (row["measure_id"] not in preferred, row["tier"], row["measure_id"]))
        row = {"plan_id": str(uuid4()), "event_id": event_id, "event_revision": event["event_revision"], "report_id": report_id,
               "session_id": context.session_id, "source_id": context.source_id, "face_id": self._scope(context),
               "point_id": event["point_id"], "cause_ids": sorted(causes), "supported_causes": labels,
               "rule_version": event["rule_version"],
               "items": [deepcopy(measure) for measure in measures],
               "experience_ids": [item["experience_id"] for item in experiences if selected.intersection(item.get("measure_ids", []))],
               "missing_fields": ([] if causes else ["cause_pending_verify"]) + ([] if measures else ["approved_applicable_measures"]),
               "state": "draft", "operation_context": "field_record" if context.mode == "realtime" else "rehearsal",
               "as_of": utcstr(context.as_of), "created_at": utcstr(now_utc()), "history": [self._audit(context, "draft_control_plan")]}
        self.store.put("control_plan", row["plan_id"], row)
        return row

    def review_plan(self, context, plan_id, decision, note="", task_due_at=None):
        self._scope(context, review=True)
        plan = self._get(context, "control_plan", plan_id)
        if decision not in ("approved", "rejected") or plan["state"] != "draft":
            raise DomainError("仅可审核draft计划", "conflict", 409)
        event = self._event(context, plan["event_id"])
        report = self._report(context, event, plan["report_id"])
        if decision == "approved" and (not plan["items"] or any(not self._applicable(context, row, report) or row.get("assigned_role") not in _ROLES for row in plan["items"])):
            raise DomainError("计划缺少当前已审核适用措施", "measure_not_approved", 409)
        due = task_due_at or {}
        if not isinstance(due, dict) or set(due) - {row["measure_id"] for row in plan["items"]}:
            raise DomainError("任务时限需要按措施编号指定")
        reviewed = now_utc()
        deadline_origin = reviewed if context.mode == "realtime" else context.as_of
        tasks = []
        if decision == "approved":
            for item in plan["items"]:
                due_at = _moment(due[item["measure_id"]]) if item["measure_id"] in due else deadline_origin + timedelta(seconds=item["deadline_seconds"]) if item.get("deadline_seconds") else None
                tasks.append({"task_id": str(uuid4()), **{key: plan[key] for key in ("plan_id", "source_id", "session_id", "face_id", "event_id", "event_revision")},
                    "measure_id": item["measure_id"], "reference_id": item["reference_id"], "text": item["text"], "assigned_role": item["assigned_role"],
                    "deadline": item["deadline"], "deadline_basis": "reviewer" if item["measure_id"] in due else "source" if due_at else "pending",
                    "due_at": utcstr(due_at) if due_at else None, "state": "assigned", "created_at": utcstr(reviewed)})
        plan.update(state=decision, reviewed_at=utcstr(reviewed), reviewed_by=context.actor.user_id)
        plan["history"].append(self._audit(context, "review_plan", note))
        with self.store.transaction() as connection:
            for task in tasks:
                self.store.put("control_task", task["task_id"], task, connection)
            self.store.put("control_plan", plan_id, plan, connection)
        return self.get_workflow(context, plan_id)

    def record_execution(self, context, task_id, content, occurred_at=None):
        task = self._get(context, "control_task", task_id)
        if task["assigned_role"] not in context.actor.roles and "safety_management" not in context.actor.roles:
            raise DomainError("当前角色不能登记该任务", "forbidden", 403)
        if not isinstance(content, str) or not content.strip() or task["state"] != "assigned":
            raise DomainError("仅可登记待执行任务且内容不能为空", "conflict", 409)
        plan = self._get(context, "control_plan", task["plan_id"])
        event = self._plan_event(context, plan)
        report = self._report(context, event, plan["report_id"])
        if plan["state"] != "approved":
            raise DomainError("计划尚未批准或事件修订已变化", "conflict", 409)
        if any(not self._applicable(context, item, report) for item in plan["items"]):
            raise DomainError("措施审核状态或引用效力已变化", "measure_not_approved", 409)
        task.update(state="executed", content=content.strip(), occurred_at=utcstr(_moment(occurred_at)) if occurred_at else utcstr(now_utc()),
                    recorded_at=utcstr(now_utc()), data_as_of=utcstr(context.as_of), actor_id=context.actor.user_id)
        self.store.put("control_task", task_id, task)
        return self.get_workflow(context, task["plan_id"])

    @staticmethod
    def _window(rows):
        good = [row for row in rows if isinstance(row.get("value"), (int, float)) and not isinstance(row["value"], bool)
                and isfinite(row["value"]) and not set(row.get("quality_flags", [])).intersection(_BAD_QUALITY)]
        values = [row["value"] for row in good]
        return {"count": len(values), "mean": sum(values) / len(values) if values else None,
                "maximum": max(values) if values else None}, good

    def verify_control_effect(self, context, plan_id):
        plan = self._get(context, "control_plan", plan_id)
        if plan["state"] == "completed":
            return self.store.get("control_effect", plan_id)
        self._report(context, self._plan_event(context, plan), plan["report_id"])
        workflow = self.get_workflow(context, plan_id)
        executed = [task for task in workflow["tasks"] if task["state"] == "executed"]
        anchor = max((_moment(task["data_as_of"]) for task in executed), default=None)
        result = {"effect_id": str(uuid4()), "plan_id": plan_id, "source_id": context.source_id, "session_id": context.session_id,
                  "face_id": plan["face_id"], "status": "insufficient_data", "semantics": "observational_comparison",
                  "window_seconds": 300, "as_of": utcstr(context.as_of), "checked_at": utcstr(now_utc()), "observation_refs": [],
                  "missing_fields": [], "escalation_candidates": []}
        if plan["state"] != "approved" or not executed or len(executed) != len(workflow["tasks"]):
            result["missing_fields"] = ["all_tasks_executed"]
        elif context.as_of < anchor + timedelta(seconds=300):
            result["missing_fields"] = ["post_execution_window"]
        else:
            windows, fragments = {}, []
            for name, start, end in (("before", anchor - timedelta(seconds=300), anchor), ("after", anchor, anchor + timedelta(seconds=300))):
                response = self.services["data"].query_series(context=context, point_id=plan["point_id"], start_at=start, end_at=end, limit=100000)
                summary, valid = self._window(response.get("rows", []))
                windows[name] = {**summary, "start_at": utcstr(start), "end_at": utcstr(end)}
                fragments.extend(valid)
            result.update(windows)
            rules = self.services["alerts"].get_rules(context.actor, plan["face_id"], plan["point_id"], plan["rule_version"])
            threshold = rules.get("observed_red_threshold")
            result["policy"] = {"threshold": threshold, "rule_status": rules.get("status"), "unit": rules.get("unit"),
                                "criterion": "after_maximum_below_threshold_and_mean_lower_than_before", "minimum_samples_per_window": 2}
            result["observation_refs"] = [{key: row[key] for key in ("id", "event_time", "source_file_id", "row_number") if key in row} for row in fragments]
            if min(windows["before"]["count"], windows["after"]["count"]) < 2 or not isinstance(threshold, (int, float)):
                result["missing_fields"] = ["valid_window_samples_or_threshold"]
            else:
                result["status"] = "effective" if windows["after"]["maximum"] < threshold and windows["after"]["mean"] < windows["before"]["mean"] else "ineffective"
            if result["status"] == "ineffective":
                current_tier = max((item["tier"] for item in plan["items"]), default=1)
                report = self._get(context, "investigation_report", plan["report_id"])
                result["escalation_candidates"] = [row for row in self.list_measures(context) if row["tier"] > current_tier
                    and set(row["cause_ids"]).intersection(plan["cause_ids"]) and self._applicable(context, row, report)]
                result["escalation_state"] = "awaiting_human_decision" if result["escalation_candidates"] else "approved_higher_tier_measures_missing"
        self.store.put("control_effect", plan_id, result)
        return result

    def review_completion(self, context, plan_id, note=""):
        self._scope(context, review=True)
        plan = self._get(context, "control_plan", plan_id)
        effect = self.store.get("control_effect", plan_id)
        if plan["state"] != "approved" or not effect or effect["status"] != "effective" or not isinstance(note, str) or not note.strip():
            raise DomainError("需要有效效果复核及非空复核说明", "conflict", 409)
        self._report(context, self._plan_event(context, plan), plan["report_id"])
        experience = {"experience_id": str(uuid4()), "plan_id": plan_id, "source_id": context.source_id, "session_id": context.session_id,
                      "face_id": plan["face_id"], "cause_ids": list(plan["cause_ids"]), "measure_ids": [row["measure_id"] for row in plan["items"]],
                      "reference_ids": list(dict.fromkeys(row["reference_id"] for row in plan["items"])), "effect": deepcopy(effect),
                      "operation_context": plan["operation_context"], "state": "pending", "history": [self._audit(context, "review_completion", note)]}
        plan.update(state="completed", completed_at=utcstr(now_utc()), experience_id=experience["experience_id"])
        plan["history"].append(self._audit(context, "review_completion", note))
        with self.store.transaction() as connection:
            self.store.put("control_experience", experience["experience_id"], experience, connection)
            self.store.put("control_plan", plan_id, plan, connection)
        return self.get_workflow(context, plan_id)

    def review_experience(self, context, experience_id, decision, note=""):
        self._scope(context, review=True)
        row = self._get(context, "control_experience", experience_id, library=True)
        if row["state"] != "pending" or decision not in ("approved", "rejected"):
            raise DomainError("仅可审核待审经验", "conflict", 409)
        for reference in row["reference_ids"]:
            self._reference(context, reference)
        row.update(state=decision, reviewed_at=utcstr(now_utc()), reviewed_by=context.actor.user_id)
        row["history"].append(self._audit(context, "review_experience", note))
        self.store.put("control_experience", experience_id, row)
        return row

    def list_plans(self, context):
        face = self._scope(context)
        return [row for row in self.store.list("control_plan") if row.get("source_id") == context.source_id
                and row.get("session_id") == context.session_id and row.get("face_id") == face]

    def process_pending_effects(self):
        result = {"processed": 0, "waiting": 0, "failed": 0}
        sessions = self.services.get("sessions")
        if sessions is None:
            return result
        for plan in self.store.list("control_plan"):
            if plan.get("state") != "approved":
                continue
            tasks = [row for row in self.store.list("control_task") if row.get("plan_id") == plan["plan_id"]]
            if not tasks or any(row["state"] != "executed" for row in tasks):
                result["waiting"] += 1
                continue
            try:
                actor = Actor(user_id="control-effect-worker", roles=("safety_management",),
                              face_ids=(plan["face_id"],), is_system=True)
                context = sessions.get_context(actor, plan["session_id"])
                if context.source_id != plan["source_id"]:
                    raise DomainError("计划与会话来源不一致", "forbidden", 403)
                previous = self.store.get("control_effect", plan["plan_id"])
                if previous and (previous["status"] in ("effective", "ineffective") or _moment(previous["as_of"]) >= context.as_of):
                    result["waiting"] += 1
                    continue
                self.verify_control_effect(context, plan["plan_id"])
                result["processed"] += 1
            except DomainError as error:
                result["failed"] += 1
                self.store.put("control_effect_worker_error", plan["plan_id"], {
                    "plan_id": plan["plan_id"], "error_code": error.code, "recorded_at": utcstr(now_utc())})
        return result

    def get_workflow(self, context, plan_id):
        plan = self._get(context, "control_plan", plan_id)
        tasks = [row for row in self.store.list("control_task") if row.get("plan_id") == plan_id]
        clock = now_utc() if context.mode == "realtime" else context.as_of
        for task in tasks:
            task["overdue"] = task["state"] != "executed" and bool(task.get("due_at")) and _moment(task["due_at"]) < clock
        return {"plan": plan, "tasks": tasks, "effect": self.store.get("control_effect", plan_id),
                "experience": self.store.get("control_experience", plan["experience_id"]) if plan.get("experience_id") else None}
