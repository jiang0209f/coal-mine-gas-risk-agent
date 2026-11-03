"""有审核状态和页段出处的资料、建议与经验。"""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext, now_utc, utcstr
from app.services.auth.service import AuthService

__all__ = ["KnowledgeService"]


class KnowledgeService:
    def __init__(self, store: Store, allowed_roots=None) -> None:
        self.store = store
        self.auth = AuthService(store)
        self.allowed_roots = [Path(path).resolve() for path in (allowed_roots or [])]

    def import_reference(self, actor: Actor, spec: dict) -> dict:
        self.auth.authorize(actor, "manage_data")
        applicability = self._applicability(actor, spec.get("applicability"))
        for field in ("title", "source", "version"):
            if not str(spec.get(field) or "").strip():
                raise DomainError(f"资料缺少{field}")
        fragments = self._extract_fragments(spec)
        document_id = str(uuid4())
        doc = {
            "document_id": document_id, "entry_id": document_id, "kind": "document",
            "version": spec["version"], "title": spec["title"], "source": spec["source"],
            "applicability": applicability, "locator": spec.get("locator"),
            "state": "pending_review", "reviewer_id": None,
            "operation_context": "field_record", "submitted_by": actor.user_id,
            "created_at": utcstr(now_utc()), "fragment_count": len(fragments),
        }
        with self.store.transaction() as connection:
            self.store.put("reference", self._key(document_id, doc["version"]), doc, connection)
            for fragment in fragments:
                self._save_fragment(doc, fragment["text"], fragment["locator"], connection)
        return doc

    def review_reference(
        self, actor: Actor, document_id: str, version, decision: str, comment: str = "",
    ) -> dict:
        self.auth.authorize(actor, "review_reference")
        return self._review(actor, "reference", document_id, version, decision, comment)

    def list_references(
        self, actor: Actor, state: str | None = None, kind: str | None = None,
        face_id: str | None = None,
    ) -> list[dict]:
        self.auth.authorize(actor, "read", face_id)
        entries = self.store.list("reference") + self.store.list("experience")
        return [entry for entry in entries if
                set(entry["applicability"]["face_ids"]).intersection(actor.face_ids)
                and (face_id is None or face_id in entry["applicability"]["face_ids"])
                and (state is None or entry["state"] == state)
                and (kind is None or entry["kind"] == kind)]

    def get_reference(self, actor: Actor, entry_id: str, version=None, locator=None) -> dict:
        self.auth.authorize(actor, "read")
        entries = [entry for entry in self.list_references(actor)
                   if entry["entry_id"] == entry_id
                   and (version is None or str(entry["version"]) == str(version))]
        if not entries:
            raise DomainError("资料版本不存在或不在授权范围", code="not_found", status_code=404)
        entry = entries[-1]
        fragments = [fragment for fragment in self.store.list("reference_fragment")
                     if fragment["entry_id"] == entry_id
                     and str(fragment["version"]) == str(entry["version"])
                     and (locator is None or fragment["locator"] == locator)]
        return {**entry, "fragments": fragments}

    def search_documents(self, context: ExecutionContext, query: str, event_scope=None) -> list[dict]:
        face_id = self._context_face(context, event_scope)
        self.auth.authorize(context.actor, "read", face_id)
        if not isinstance(query, str):
            raise DomainError("检索词必须为字符串")
        found = []
        for fragment in self.store.list("reference_fragment"):
            if query.strip() and query.casefold() not in fragment["text"].casefold():
                continue
            try:
                found.append(self._approved_fragment(
                    fragment["reference_id"], face_id, context.mode == "realtime"
                ))
            except DomainError:
                continue
        return found[:50]

    def draft_advice(self, context: ExecutionContext, evidence_bundle: dict, reference_ids: list[str]) -> dict:
        event = evidence_bundle.get("event") or evidence_bundle.get("event_snapshot")
        if not isinstance(event, dict) or not event.get("event_id"):
            raise DomainError("建议需要实际事件依据包")
        if event.get("session_id") != context.session_id:
            raise DomainError("事件不属于当前会话", code="forbidden", status_code=403)
        face_id = self._context_face(context, event)
        self.auth.authorize(context.actor, "draft_advice", face_id)
        if not isinstance(reference_ids, list) or not reference_ids:
            raise DomainError("建议需要非空资料引用")
        fragments = [self._approved_fragment(ref, face_id, context.mode == "realtime")
                     for ref in dict.fromkeys(reference_ids)]
        with self.store.transaction() as connection:
            previous = [entry for entry in self.store.list("advice", connection)
                        if entry["event_id"] == event["event_id"]
                        and entry["session_id"] == context.session_id]
            advice_id = previous[0]["advice_id"] if previous else str(uuid4())
            version = max((entry["version"] for entry in previous), default=0) + 1
            advice = {
                "advice_id": advice_id, "version": version, "event_id": event["event_id"],
                "event_revision": event["event_revision"], "session_id": context.session_id,
                "face_id": face_id, "applicability": {"face_ids": [face_id]},
                "reference_ids": [fragment["reference_id"] for fragment in fragments],
                "items": [{"text": fragment["text"], "reference_id": fragment["reference_id"],
                           "source": fragment["source"], "locator": fragment["locator"]}
                          for fragment in fragments],
                "operation_context": "field_record" if context.mode == "realtime" else "rehearsal",
                "created_at": utcstr(now_utc()), "data_as_of": utcstr(context.as_of),
            }
            self.store.put("advice", self._key(advice_id, version), advice, connection)
        return advice

    def get_advice(self, actor: Actor, advice_id: str, version=1) -> dict:
        advice = self.store.get("advice", self._key(advice_id, version))
        if advice is None:
            raise DomainError("建议版本不存在", code="not_found", status_code=404)
        self.auth.authorize(actor, "read", advice["face_id"])
        return advice

    def list_advice(self, actor: Actor, event_id: str) -> list[dict]:
        self.auth.authorize(actor, "read")
        return sorted([entry for entry in self.store.list("advice")
                       if entry["event_id"] == event_id and entry["face_id"] in actor.face_ids],
                      key=lambda entry: entry["version"])

    def submit_experience(self, actor: Actor, reviewed_case_snapshot: dict, spec: dict) -> dict:
        self.auth.authorize(actor, "submit_feedback")
        required = ("case_id", "event_id", "review_action_ref", "reviewed_at", "snapshot_ref", "face_id")
        snapshot = reviewed_case_snapshot
        if not isinstance(snapshot, dict) or any(not snapshot.get(key) for key in required):
            raise DomainError("经验需要入口提供的已复核处置快照")
        operation_context = snapshot.get("operation_context")
        if operation_context not in ("rehearsal", "field_record"):
            raise DomainError("处置来源性质无效")
        actor.require_face(snapshot["face_id"])
        applicability = self._applicability(actor, spec.get("applicability"))
        if not isinstance(spec.get("content"), str) or not spec["content"].strip():
            raise DomainError("经验内容不能为空")
        refs = list(dict.fromkeys(snapshot.get("source_refs", []) + spec.get("source_refs", [])))
        for reference_id in refs:
            self._approved_fragment(reference_id, snapshot["face_id"], operation_context == "field_record")
        experience_id = str(uuid4())
        experience = {
            "experience_id": experience_id, "entry_id": experience_id, "kind": "experience",
            "version": 1, "source_case_id": snapshot["case_id"], "content": spec["content"],
            "source": f"处置单 {snapshot['case_id']}", "applicability": applicability,
            "source_refs": refs, "state": "pending_review", "submitted_by": actor.user_id,
            "reviewed_by": None, "operation_context": operation_context,
            "source_case_snapshot_ref": snapshot["snapshot_ref"], "source_case_snapshot": snapshot,
        }
        with self.store.transaction() as connection:
            self.store.put("experience", self._key(experience_id, 1), experience, connection)
            self._save_fragment(experience, spec["content"], {"case_id": snapshot["case_id"]}, connection)
        return experience

    def review_experience(self, actor: Actor, experience_id: str, version, decision: str, comment: str = "") -> dict:
        self.auth.authorize(actor, "review_experience")
        return self._review(actor, "experience", experience_id, version, decision, comment)

    def _review(self, actor, kind, entry_id, version, decision, comment) -> dict:
        if decision not in ("approved", "rejected"):
            raise DomainError("审核结论必须为approved或rejected")
        with self.store.transaction() as connection:
            entry = self.store.get(kind, self._key(entry_id, version), connection)
            if entry is None:
                raise DomainError("待审核版本不存在", code="not_found", status_code=404)
            self._applicability(actor, entry["applicability"])
            if entry["state"] != "pending_review":
                raise DomainError("该版本已经审核", code="conflict", status_code=409)
            entry.update(state=decision, reviewer_id=actor.user_id, reviewed_by=actor.user_id,
                         review_comment=comment, reviewed_at=utcstr(now_utc()))
            self.store.put(kind, self._key(entry_id, version), entry, connection)
        return entry

    def _approved_fragment(self, reference_id, face_id, field_record=False) -> dict:
        fragment = self.store.get("reference_fragment", reference_id)
        if fragment is None:
            raise DomainError("引用资料不存在")
        kind = "reference" if fragment["kind"] == "document" else "experience"
        entry = self.store.get(kind, self._key(fragment["entry_id"], fragment["version"]))
        if entry is None or entry["state"] != "approved":
            raise DomainError("引用资料尚未批准")
        if face_id not in fragment["applicability"]["face_ids"]:
            raise DomainError("引用资料不适用于当前工作面", code="forbidden", status_code=403)
        if field_record and fragment["operation_context"] == "rehearsal":
            raise DomainError("现场建议不能使用演练经验", code="forbidden", status_code=403)
        return fragment

    def _context_face(self, context, scope=None) -> str:
        source = self.store.get("source", context.source_id)
        if source is None:
            raise DomainError("当前数据源不存在", code="not_found", status_code=404)
        face_id = source["face_id"]
        if scope and scope.get("event_id"):
            if scope.get("session_id") != context.session_id or scope.get("source_id") != context.source_id:
                raise DomainError("事件不属于当前会话或数据源", code="forbidden", status_code=403)
            try:
                revision_as_of = datetime.fromisoformat(scope["revision_as_of"])
                utcstr(revision_as_of)
            except (KeyError, ValueError, TypeError) as exc:
                raise DomainError("事件缺少有效的修订时间") from exc
            if revision_as_of > context.as_of:
                raise DomainError("事件修订在当前数据时间尚不可见", code="conflict", status_code=409)
            point = self.store.get("point", scope.get("point_id", ""))
            if point is not None and point.get("source_id") != context.source_id:
                raise DomainError("事件测点不属于当前数据源", code="forbidden", status_code=403)
        if scope and scope.get("face_id", face_id) != face_id:
            raise DomainError("资料范围与会话工作面不一致", code="forbidden", status_code=403)
        return face_id

    @staticmethod
    def _applicability(actor, applicability) -> dict:
        faces = applicability.get("face_ids") if isinstance(applicability, dict) else None
        if not isinstance(faces, list) or not faces or any(not isinstance(face, str) or not face for face in faces):
            raise DomainError("适用范围需要非空face_ids列表")
        for face_id in faces:
            actor.require_face(face_id)
        return {**applicability, "face_ids": list(dict.fromkeys(faces))}

    def _save_fragment(self, entry, text, locator, connection) -> None:
        reference_id = str(uuid4())
        self.store.put("reference_fragment", reference_id, {
            "reference_id": reference_id, "entry_id": entry["entry_id"], "kind": entry["kind"],
            "version": entry["version"], "locator": locator, "text": text,
            "source": entry["source"], "applicability": entry["applicability"],
            "operation_context": entry["operation_context"],
        }, connection)

    def _extract_fragments(self, spec) -> list[dict]:
        fragments = spec.get("fragments")
        if fragments is None and spec.get("text"):
            fragments = [{"text": spec["text"], "locator": spec.get("locator") or {"paragraph": 1}}]
        if fragments is None and spec.get("file_ref"):
            path = Path(spec["file_ref"]).resolve()
            if not any(path.is_relative_to(root) for root in self.allowed_roots):
                raise DomainError("资料文件不在允许的导入目录", code="forbidden", status_code=403)
            try:
                if path.suffix.lower() in (".txt", ".md"):
                    fragments = [{"text": line, "locator": {"line": index}}
                                 for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1) if line.strip()]
                elif path.suffix.lower() == ".pdf":
                    from pypdf import PdfReader
                    fragments = [{"text": page.extract_text() or "", "locator": {"page": index}}
                                 for index, page in enumerate(PdfReader(path).pages, 1)]
                    fragments = [fragment for fragment in fragments if fragment["text"].strip()]
                elif path.suffix.lower() == ".docx":
                    from docx import Document
                    fragments = [{"text": paragraph.text, "locator": {"paragraph": index}}
                                 for index, paragraph in enumerate(Document(path).paragraphs, 1) if paragraph.text.strip()]
                else:
                    raise DomainError("支持导入txt、md、pdf和docx资料")
            except (OSError, ValueError):
                raise DomainError("资料文件读取失败") from None
        if not isinstance(fragments, list) or not fragments or any(
            not isinstance(fragment, dict) or not isinstance(fragment.get("text"), str)
            or not fragment["text"].strip() or not fragment.get("locator") for fragment in fragments
        ):
            raise DomainError("资料需要非空原文和页段定位")
        return fragments

    @staticmethod
    def _key(entry_id, version) -> str:
        return f"{entry_id}:{version}"
