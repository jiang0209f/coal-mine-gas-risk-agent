"""有审核状态和页段出处的资料、建议与经验。"""

from datetime import datetime
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from uuid import uuid4
from zipfile import ZipFile

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext, now_utc, utcstr
from app.services.auth.service import AuthService
from app.services.knowledge.conflicts import annotate_conflicts
from app.services.knowledge.retrieval import retrieve

__all__ = ["KnowledgeService"]

_MAX_FILE_BYTES = 20 * 1024 * 1024
_MEDIA_TYPES = {".txt": "text/plain; charset=utf-8", ".md": "text/markdown; charset=utf-8",
                ".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


class KnowledgeService:
    def __init__(self, store: Store, allowed_roots=None, upload_dir=None, clock=now_utc, embedding=None) -> None:
        self.store = store
        self.auth = AuthService(store)
        self.allowed_roots = [Path(path).resolve() for path in (allowed_roots or [])]
        self.upload_dir = Path(upload_dir).resolve() if upload_dir is not None else None
        self.clock = clock
        self.embedding = embedding

    def upload_file(self, actor: Actor, filename: str, content: bytes, face_ids=None) -> dict:
        self.auth.authorize(actor, "manage_data")
        faces = self._applicability(actor, {"face_ids": list(actor.face_ids) if face_ids is None else face_ids})["face_ids"]
        if self.upload_dir is None:
            raise DomainError("资料上传目录尚未配置", code="service_unavailable", status_code=503)
        if (not isinstance(filename, str) or not filename.strip() or len(filename) > 200
                or any(character in filename for character in ("/", "\\", ":"))
                or any(ord(character) < 32 or ord(character) == 127 for character in filename)):
            raise DomainError("资料文件名无效")
        extension = Path(filename).suffix.lower()
        if extension not in _MEDIA_TYPES:
            raise DomainError("支持上传PDF、DOCX、TXT和MD资料")
        if not isinstance(content, bytes) or not content:
            raise DomainError("资料文件不能为空")
        if len(content) > _MAX_FILE_BYTES:
            raise DomainError("资料文件不能超过20MiB", code="file_too_large", status_code=413)
        self._validate_content(extension, content)
        entry = {"file_id": str(uuid4()), "filename": filename, "extension": extension,
                 "media_type": _MEDIA_TYPES[extension], "size_bytes": len(content),
                 "sha256": hashlib.sha256(content).hexdigest(), "face_ids": faces,
                 "uploaded_by": actor.user_id, "created_at": utcstr(now_utc())}
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        path = self._file_path(entry)
        try:
            with path.open("xb") as output:
                output.write(content)
            self.store.put("knowledge_file", entry["file_id"], entry)
        except OSError:
            raise DomainError("资料文件保存失败", code="file_storage_failed", status_code=503) from None
        return entry

    @staticmethod
    def _validate_content(extension, content):
        try:
            if extension in (".txt", ".md"):
                if not content.decode("utf-8-sig").strip() or b"\x00" in content:
                    raise ValueError("empty text")
            elif extension == ".pdf":
                from pypdf import PdfReader
                if not content.startswith(b"%PDF-"):
                    raise ValueError("PDF signature")
                reader = PdfReader(BytesIO(content))
                if reader.is_encrypted or not reader.pages:
                    raise ValueError("PDF pages")
            else:
                with ZipFile(BytesIO(content)) as archive:
                    members = archive.infolist()
                    if (len(members) > 1000 or sum(item.file_size for item in members) > 64 * 1024 * 1024
                            or not {"[Content_Types].xml", "word/document.xml"} <= set(archive.namelist())):
                        raise ValueError("DOCX package")
        except Exception:
            raise DomainError("资料格式无效，TXT和MD需要UTF-8文本") from None

    def _file_path(self, entry):
        if self.upload_dir is None:
            raise DomainError("资料上传目录尚未配置", code="service_unavailable", status_code=503)
        path = (self.upload_dir / (entry["file_id"] + entry["extension"])).resolve()
        if not path.is_relative_to(self.upload_dir):
            raise DomainError("资料文件不在本地存储目录", code="forbidden", status_code=403)
        return path

    def get_file(self, actor: Actor, file_id: str) -> dict:
        self.auth.authorize(actor, "read")
        if not isinstance(file_id, str) or not file_id.strip():
            raise DomainError("资料文件编号无效")
        entry = self.store.get("knowledge_file", file_id)
        if entry is None:
            raise DomainError("资料文件不存在", code="not_found", status_code=404)
        for face in entry["face_ids"]:
            actor.require_face(face)
        return entry

    def get_file_content(self, actor: Actor, file_id: str) -> tuple[dict, bytes]:
        entry = self.get_file(actor, file_id)
        try:
            with self._file_path(entry).open("rb") as original:
                content = original.read(_MAX_FILE_BYTES + 1)
        except OSError:
            raise DomainError("资料原件读取失败", code="file_unavailable", status_code=404) from None
        if len(content) != entry["size_bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise DomainError("资料原件摘要不一致", code="file_integrity_failed", status_code=409)
        return entry, content

    def import_reference(self, actor: Actor, spec: dict) -> dict:
        self.auth.authorize(actor, "manage_data")
        applicability = self._applicability(actor, spec.get("applicability"))
        for field in ("title", "source", "version"):
            if not str(spec.get(field) or "").strip():
                raise DomainError(f"资料缺少{field}")
        uploaded = None
        if spec.get("file_id"):
            if any(spec.get(key) is not None for key in ("text", "fragments", "file_ref")):
                raise DomainError("上传文件不能与内联原文或文件路径同时导入")
            uploaded, _ = self.get_file_content(actor, spec["file_id"])
            if uploaded["uploaded_by"] != actor.user_id:
                raise DomainError("资料文件不属于当前上传人", code="forbidden", status_code=403)
            if not set(applicability["face_ids"]).issubset(uploaded["face_ids"]):
                raise DomainError("资料适用范围超出上传文件工作面", code="forbidden", status_code=403)
            fragments = self._read_fragments(self._file_path(uploaded))
        else:
            fragments = self._extract_fragments(spec)
        fragments = self._chunk_fragments(fragments)
        requested = spec.get("entry_id", spec.get("document_id"))
        if spec.get("entry_id") is not None and spec.get("document_id") is not None and spec["entry_id"] != spec["document_id"]:
            raise DomainError("资料编号不一致")
        if requested is not None and (not isinstance(requested, str) or not requested.strip()):
            raise DomainError("资料编号无效")
        document_id = requested or str(uuid4())
        doc = {
            "document_id": document_id, "entry_id": document_id, "kind": "document",
            "version": spec["version"], "title": spec["title"], "source": spec["source"],
            "applicability": applicability, "locator": spec.get("locator"),
            "state": "pending_review", "reviewer_id": None,
            "operation_context": "field_record", "submitted_by": actor.user_id,
            "created_at": utcstr(now_utc()), "fragment_count": len(fragments),
        }
        if uploaded is not None:
            doc.update(file_id=uploaded["file_id"], file_sha256=uploaded["sha256"], filename=uploaded["filename"])
        with self.store.transaction() as connection:
            previous = [entry for entry in self.store.list("reference", connection) if entry["entry_id"] == document_id]
            if requested and not previous:
                raise DomainError("待修订资料不存在", code="not_found", status_code=404)
            for entry in previous:
                self._applicability(actor, entry["applicability"])
            if self.store.get("reference", self._key(document_id, doc["version"]), connection) is not None:
                raise DomainError("该资料版本已存在，请使用新版本号", code="conflict", status_code=409)
            doc["revision_order"] = max((entry.get("revision_order", 0) for entry in previous), default=0) + 1
            self.store.put("reference", self._key(document_id, doc["version"]), doc, connection)
            for index, fragment in enumerate(fragments, 1):
                self._save_fragment(doc, fragment["text"], fragment["locator"], connection, index)
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
        policies = {self._key(row["entry_id"], row["version"]): row for row in self.store.list("knowledge_disclosure")}
        return [{**entry, "disclosure": policies.get(self._key(entry["entry_id"], entry["version"]), {}).get("disclosure", "local_only")} for entry in entries if
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
        if version is None:
            current = []
            knowledge_as_of = self.clock()
            for candidate in entries:
                if candidate["state"] != "approved":
                    continue
                try:
                    self._require_effective(candidate["applicability"], knowledge_as_of)
                    current.append(candidate)
                except DomainError:
                    continue
            entries = current or entries
        entry = max(entries, key=lambda item: (item.get("revision_order", 0), item.get("created_at", ""), str(item["version"])))
        fragments = [fragment for fragment in self.store.list("reference_fragment")
                     if fragment["entry_id"] == entry_id
                     and str(fragment["version"]) == str(entry["version"])
                     and (locator is None or fragment["locator"] == locator)]
        conflicts = [record for record in self.store.list("knowledge_conflict")
                     if set(record.get("face_ids", [])).intersection(actor.face_ids)]
        conflicts = self._effective_conflicts(conflicts, set(entry["applicability"]["face_ids"]).intersection(actor.face_ids))
        fragments = annotate_conflicts(fragments, {"conflicts": conflicts})
        return {**entry, "fragments": sorted(fragments, key=lambda item: item.get("ordinal", 0))}

    def search_documents(self, context: ExecutionContext, query: str, event_scope=None) -> list[dict]:
        return self._search(context, query, event_scope)

    def search_public_documents(self, context: ExecutionContext, query: str, event_scope=None) -> list[dict]:
        return self._search(context, query, event_scope, public_only=True)

    def _candidates(self, context, event_scope=None, public_only=False):
        face_id = self._context_face(context, event_scope)
        self.auth.authorize(context.actor, "read", face_id)
        knowledge_as_of = self.clock()
        documents = self.store.list("reference") + self.store.list("experience")
        fragments = self.store.list("reference_fragment")
        grouped = {}
        for fragment in fragments:
            grouped.setdefault(self._key(fragment["entry_id"], fragment["version"]), []).append(fragment)
        policies = {self._key(row["entry_id"], row["version"]): row for row in self.store.list("knowledge_disclosure")} if public_only else {}
        current = {}
        for entry in documents:
            if entry["state"] != "approved" or face_id not in entry["applicability"]["face_ids"]:
                continue
            try:
                self._require_effective(entry["applicability"], knowledge_as_of)
            except DomainError:
                continue
            if context.mode == "realtime" and entry.get("operation_context") == "rehearsal":
                continue
            key = self._key(entry["entry_id"], entry["version"])
            if public_only and not self._valid_public_policy(entry, grouped.get(key, []), policies.get(key)):
                continue
            current[key] = entry
        return [{**fragment, "title": current[key].get("title") or "案例经验", "knowledge_as_of": utcstr(knowledge_as_of)}
                for fragment in fragments if (key := self._key(fragment["entry_id"], fragment["version"])) in current
                and face_id in fragment["applicability"]["face_ids"]]

    def _search(self, context, query, event_scope=None, public_only=False):
        if not isinstance(query, str) or len(query) > 20000:
            raise DomainError("检索词须为不超过20000字的字符串")
        candidates = self._candidates(context, event_scope, public_only)
        metadata = self._conflict_metadata(context, public_only)
        candidates = annotate_conflicts(candidates, metadata)
        try:
            return retrieve(query, candidates, embedding=self.embedding)
        except ValueError as error:
            raise DomainError("检索参数或索引无效") from error

    @staticmethod
    def _official_url(value):
        try:
            parsed = urlsplit(value)
            return (parsed.scheme == "https" and parsed.hostname is not None
                    and (parsed.hostname == "gov.cn" or parsed.hostname.endswith(".gov.cn"))
                    and parsed.username is None and parsed.password is None
                    and parsed.port in (None, 443) and not parsed.fragment)
        except (ValueError, TypeError, AttributeError):
            return False

    @staticmethod
    def _body_digest(entry, fragments):
        value = {"title": entry.get("title"), "source": entry["source"], "version": entry["version"],
                 "fragments": [{key: row.get(key) for key in ("reference_id", "text", "locator", "ordinal")}
                               for row in sorted(fragments, key=lambda row: row["reference_id"])]}
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _valid_public_policy(self, entry, fragments, policy):
        try:
            self._require_effective(entry["applicability"], self.clock())
        except DomainError:
            return False
        return bool(entry.get("state") == "approved" and policy and entry.get("kind") == "document" and policy.get("disclosure") == "public_normative"
                    and policy.get("category") in ("law", "regulation", "standard")
                    and self._official_url(policy.get("official_url")) and fragments
                    and policy.get("body_sha256") == self._body_digest(entry, fragments))

    def set_disclosure(self, actor, entry_id, version, disclosure, *, official_url=None, category=None, comment=""):
        self.auth.authorize(actor, "manage_data")
        if disclosure not in ("local_only", "public_normative") or not isinstance(comment, str) or not comment.strip():
            raise DomainError("外发分类须为local_only或public_normative并记录核验说明")
        document = self.get_reference(actor, entry_id, version)
        self._applicability(actor, document["applicability"])
        if disclosure == "public_normative":
            if (document.get("kind") != "document" or document["state"] != "approved"
                    or category not in ("law", "regulation", "standard") or not self._official_url(official_url)):
                raise DomainError("仅已批准且有HTTPS官方出处的公开法规标准可标为公开", code="forbidden", status_code=403)
            self._require_effective(document["applicability"], self.clock())
        policy = {"entry_id": entry_id, "version": version, "disclosure": disclosure, "official_url": official_url,
                  "category": category, "comment": comment, "reviewed_by": actor.user_id,
                  "reviewed_at": utcstr(self.clock()), "body_sha256": self._body_digest(document, document["fragments"])}
        self.store.put("knowledge_disclosure", self._key(entry_id, version), policy)
        return policy

    @staticmethod
    def _public_locator(locator):
        result = {}
        for key in ("page", "physical_page", "paragraph", "table", "row", "line", "chunk", "start_offset", "end_offset"):
            if type(locator.get(key)) is int and locator[key] >= 0:
                result[key] = locator[key]
        if isinstance(locator.get("clause"), str) and re.fullmatch(r"[第〇一二三四五六七八九十百千万\d.．条章节款项()（）-]{1,80}", locator["clause"]):
            result["clause"] = locator["clause"]
        return result

    def approved_public_fragment(self, context, reference_id):
        candidates = self._candidates(context, public_only=True)
        row = next((item for item in candidates if item["reference_id"] == reference_id), None)
        if row is None:
            raise DomainError("引用不是已批准、有效且授权公开的法规标准", code="public_reference_unavailable", status_code=403)
        annotated = next(item for item in annotate_conflicts(candidates, self._conflict_metadata(context, True)) if item["reference_id"] == reference_id)
        policy = self.store.get("knowledge_disclosure", self._key(row["entry_id"], row["version"]))
        # Only public citation locators and explicit public conflict membership
        # cross the boundary; local paths, scope, provenance and notes stay local.
        conflicts = [{"conflict_id": hashlib.sha256(item["conflict_id"].encode()).hexdigest()[:16],
                      "status": item["status"], "reference_ids": [ref["reference_id"] for ref in item["references"] if ref.get("reference_id")]}
                     for item in annotated["conflicts"]]
        return {"reference_id": reference_id, "entry_id": row["entry_id"], "document_id": row["entry_id"], "title": row["title"], "version": row["version"],
                "text": row["text"], "locator": self._public_locator(row["locator"]),
                "source": policy["official_url"], "official_url": policy["official_url"], "state": "approved", "effective": True,
                "disclosure": "public_normative", "conflict_status": annotated["conflict_status"], "conflicts": conflicts}

    def _conflict_metadata(self, context, public_only=False):
        face_id = self._context_face(context)
        records = [record for record in self.store.list("knowledge_conflict") if face_id in record.get("face_ids", [])]
        records = self._effective_conflicts(records, {face_id})
        if public_only:
            public_rows = self._candidates(context, public_only=True)
            public_ids = {row["reference_id"] for row in public_rows}
            records = [record for record in records if record.get("disclosure") == "public_normative"
                       and set(record.get("reference_ids", [])) <= public_ids
                       and (not record.get("resolution_reference") or record["resolution_reference"].get("reference_id") in public_ids)]
        return {"conflicts": records, "replacements": []}

    def _effective_conflicts(self, records, face_ids):
        result = deepcopy(records)
        for record in result:
            if record.get("status") == "resolved":
                for face_id in set(record.get("face_ids", [])).intersection(face_ids):
                    try:
                        self._approved_fragment(record["resolution_reference"].get("reference_id"), face_id)
                    except DomainError:
                        record.update(status="open", reason=record["reason"]+"；解决依据当前已失效或不适用，须重新核对。")
                        break
        return result

    def set_conflict(self, actor, record):
        self.auth.authorize(actor, "manage_data")
        if not isinstance(record, dict) or not isinstance(record.get("reference_ids"), list):
            raise DomainError("冲突登记须明确绑定引用编号数组")
        value = deepcopy(record)
        references = []
        for reference_id in value["reference_ids"]:
            fragment = self.store.get("reference_fragment", reference_id)
            if fragment is None:
                raise DomainError("冲突引用不存在")
            entry = self.get_reference(actor, fragment["entry_id"], fragment["version"])
            self._applicability(actor, entry["applicability"])
            references.append((entry, fragment))
        faces = set.intersection(*(set(entry["applicability"]["face_ids"]) for entry, _ in references)) if references else set()
        if not faces:
            raise DomainError("冲突条文须有共同适用工作面")
        value.setdefault("disclosure", "local_only")
        if value["disclosure"] not in ("local_only", "public_normative"):
            raise DomainError("冲突外发分类无效")
        if value["disclosure"] == "public_normative":
            for entry, _ in references:
                policy = self.store.get("knowledge_disclosure", self._key(entry["entry_id"], entry["version"]))
                if not self._valid_public_policy(entry, entry["fragments"], policy):
                    raise DomainError("公开冲突的所有参与资料须已核准公开", code="forbidden", status_code=403)
        if value.get("status") == "resolved":
            resolution = value.get("resolution_reference", {}).get("reference_id")
            if not resolution:
                raise DomainError("解决冲突须登记核验依据引用")
            for face_id in faces:
                fragment = self._approved_fragment(resolution, face_id)
            resolved_doc = self.get_reference(actor, fragment["entry_id"], fragment["version"])
            if value["disclosure"] == "public_normative":
                policy = self.store.get("knowledge_disclosure", self._key(fragment["entry_id"], fragment["version"]))
                if not self._valid_public_policy(resolved_doc, resolved_doc["fragments"], policy):
                    raise DomainError("公开冲突的解决依据须已核准公开", code="forbidden", status_code=403)
        try:
            annotate_conflicts([], {"conflicts": [value]})
        except (ValueError, TypeError, KeyError) as error:
            raise DomainError("冲突登记格式无效") from error
        value.update(face_ids=sorted(faces), registered_by=actor.user_id, updated_at=utcstr(self.clock()))
        with self.store.transaction() as connection:
            previous = self.store.get("knowledge_conflict", value["conflict_id"], connection)
            if previous:
                for face_id in previous["face_ids"]:
                    actor.require_face(face_id)
            self.store.put("knowledge_conflict", value["conflict_id"], value, connection)
        return value

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
        knowledge_as_of = self.clock()
        fragments = [self._approved_fragment(ref, face_id, context.mode == "realtime", knowledge_as_of)
                     for ref in dict.fromkeys(reference_ids)]
        checked = annotate_conflicts(fragments, self._conflict_metadata(context))
        if any(row["conflict_status"] in ("open", "possible_conflict", "superseded") for row in checked):
            raise DomainError("引用条文有未解决分歧或已被替代，请先核对登记的双方原文和签批依据", code="knowledge_conflict_unresolved", status_code=409)
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
                           "source": fragment["source"], "locator": fragment["locator"],
                           "entry_id": fragment["entry_id"], "version": fragment["version"], "kind": fragment["kind"],
                           **({"file_id": fragment["file_id"]} if fragment.get("file_id") else {})}
                          for fragment in fragments],
                "operation_context": "field_record" if context.mode == "realtime" else "rehearsal",
                "created_at": utcstr(now_utc()), "data_as_of": utcstr(context.as_of),
                "knowledge_as_of": utcstr(knowledge_as_of),
            }
            self.store.put("advice", self._key(advice_id, version), advice, connection)
        return advice

    def get_advice(self, actor: Actor, advice_id: str, version=1) -> dict:
        advice = self.store.get("advice", self._key(advice_id, version))
        if advice is None:
            raise DomainError("建议版本不存在", code="not_found", status_code=404)
        self.auth.authorize(actor, "read", advice["face_id"])
        return advice

    def validate_advice_references(self, context: ExecutionContext, advice: dict) -> None:
        face_id = self._context_face(context)
        self.auth.authorize(context.actor, "read", face_id)
        if advice.get("face_id", face_id) != face_id:
            raise DomainError("建议资料范围与当前工作面不一致", code="forbidden", status_code=403)
        references = advice.get("reference_ids")
        if not isinstance(references, list) or not references or any(not isinstance(ref, str) or not ref for ref in references):
            raise DomainError("建议需要非空有效资料引用")
        knowledge_as_of = self.clock()
        fragments = [self._approved_fragment(reference_id, face_id, context.mode == "realtime", knowledge_as_of)
                     for reference_id in dict.fromkeys(references)]
        checked = annotate_conflicts(fragments, self._conflict_metadata(context))
        if any(row["conflict_status"] in ("open", "possible_conflict", "superseded") for row in checked):
            raise DomainError("建议引用条文出现未解决分歧或已被替代，请重新核验依据", code="knowledge_conflict_unresolved", status_code=409)

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
        if not set(applicability["face_ids"]).issubset({snapshot["face_id"]}):
            raise DomainError("经验适用范围必须保留在原处置工作面", code="forbidden", status_code=403)
        if not isinstance(spec.get("content"), str) or not spec["content"].strip():
            raise DomainError("经验内容不能为空")
        refs = list(dict.fromkeys(snapshot.get("source_refs", []) + spec.get("source_refs", [])))
        knowledge_as_of = self.clock()
        for reference_id in refs:
            self._approved_fragment(reference_id, snapshot["face_id"], operation_context == "field_record", knowledge_as_of)
        experience_id = str(uuid4())
        experience = {
            "experience_id": experience_id, "entry_id": experience_id, "kind": "experience",
            "version": 1, "source_case_id": snapshot["case_id"], "content": spec["content"],
            "source": f"处置单 {snapshot['case_id']}", "applicability": applicability,
            "source_refs": refs, "state": "pending_review", "submitted_by": actor.user_id,
            "reviewed_by": None, "operation_context": operation_context,
            "source_case_snapshot_ref": snapshot["snapshot_ref"], "source_case_snapshot": snapshot,
            "knowledge_as_of": utcstr(knowledge_as_of),
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
        knowledge_as_of = self.clock()
        with self.store.transaction() as connection:
            entry = self.store.get(kind, self._key(entry_id, version), connection)
            if entry is None:
                raise DomainError("待审核版本不存在", code="not_found", status_code=404)
            self._applicability(actor, entry["applicability"])
            if entry["state"] != "pending_review":
                raise DomainError("该版本已经审核", code="conflict", status_code=409)
            if decision == "approved":
                self._require_effective(entry["applicability"], knowledge_as_of)
                if kind == "reference":
                    siblings = [item for item in self.store.list("reference", connection)
                                if item["entry_id"] == entry_id and item["state"] == "approved"]
                    if any(item.get("revision_order", 0) > entry.get("revision_order", 0) for item in siblings):
                        raise DomainError("已有更新的批准版本，当前修订不能替代新版", code="conflict", status_code=409)
                    for previous in siblings:
                        self._applicability(actor, previous["applicability"])
                        previous.update(state="superseded", superseded_by_version=entry["version"],
                                        superseded_at=utcstr(knowledge_as_of))
                        self.store.put(kind, self._key(entry_id, previous["version"]), previous, connection)
            entry.update(state=decision, reviewer_id=actor.user_id, reviewed_by=actor.user_id,
                         review_comment=comment, reviewed_at=utcstr(knowledge_as_of))
            self.store.put(kind, self._key(entry_id, version), entry, connection)
        return entry

    def _approved_fragment(self, reference_id, face_id, field_record=False, as_of=None) -> dict:
        fragment = self.store.get("reference_fragment", reference_id)
        if fragment is None:
            raise DomainError("引用资料不存在")
        kind = "reference" if fragment["kind"] == "document" else "experience"
        entry = self.store.get(kind, self._key(fragment["entry_id"], fragment["version"]))
        if entry is None or entry["state"] != "approved":
            raise DomainError("引用资料尚未批准")
        if face_id not in fragment["applicability"]["face_ids"] or face_id not in entry["applicability"]["face_ids"]:
            raise DomainError("引用资料不适用于当前工作面", code="forbidden", status_code=403)
        if field_record and fragment["operation_context"] == "rehearsal":
            raise DomainError("现场建议不能使用演练经验", code="forbidden", status_code=403)
        current = as_of if as_of is not None else self.clock()
        self._require_effective(entry["applicability"], current)
        self._require_effective(fragment["applicability"], current)
        return fragment

    @staticmethod
    def _require_effective(applicability, knowledge_as_of):
        utcstr(knowledge_as_of)
        valid_from, valid_to = applicability.get("valid_from"), applicability.get("valid_to")
        if ((valid_from and knowledge_as_of < datetime.fromisoformat(valid_from))
                or (valid_to and knowledge_as_of >= datetime.fromisoformat(valid_to))):
            raise DomainError("资料在当前效力核验时点尚未生效或已失效", code="reference_not_effective", status_code=409)

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
        result = {**applicability, "face_ids": list(dict.fromkeys(faces))}
        for key in ("valid_from", "valid_to"):
            if result.get(key) is not None:
                try:
                    result[key] = utcstr(datetime.fromisoformat(result[key].replace("Z", "+00:00")))
                except (ValueError, TypeError, AttributeError):
                    raise DomainError("资料有效时间需要带时区的ISO格式") from None
        if result.get("valid_from") and result.get("valid_to"):
            if datetime.fromisoformat(result["valid_from"]) >= datetime.fromisoformat(result["valid_to"]):
                raise DomainError("资料失效时间必须晚于生效时间")
        return result

    def _save_fragment(self, entry, text, locator, connection, ordinal=1) -> None:
        reference_id = str(uuid4())
        self.store.put("reference_fragment", reference_id, {
            "reference_id": reference_id, "entry_id": entry["entry_id"], "kind": entry["kind"],
            "version": entry["version"], "locator": locator, "text": text,
            "source": entry["source"], "applicability": entry["applicability"],
            "operation_context": entry["operation_context"],
            "ordinal": ordinal,
            **({"file_id": entry["file_id"], "file_sha256": entry["file_sha256"]} if entry.get("file_id") else {}),
        }, connection)

    def _extract_fragments(self, spec) -> list[dict]:
        fragments = spec.get("fragments")
        if fragments is None and spec.get("text"):
            fragments = [{"text": spec["text"], "locator": spec.get("locator") or {"paragraph": 1}}]
        if fragments is None and spec.get("file_ref"):
            path = Path(spec["file_ref"]).resolve()
            if not any(path.is_relative_to(root) for root in self.allowed_roots):
                raise DomainError("资料文件不在允许的导入目录", code="forbidden", status_code=403)
            fragments = self._read_fragments(path)
        if not isinstance(fragments, list) or not fragments or any(
            not isinstance(fragment, dict) or not isinstance(fragment.get("text"), str)
            or not fragment["text"].strip() or not fragment.get("locator") for fragment in fragments
        ):
            raise DomainError("资料需要非空原文和页段定位")
        return fragments

    @staticmethod
    def _read_fragments(path):
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                raise DomainError("资料文件不能超过20MiB", code="file_too_large", status_code=413)
            extension = path.suffix.lower()
            if extension in (".txt", ".md"):
                return [{"text": line, "locator": {"line": index}}
                        for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1) if line.strip()]
            if extension == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(path)
                if reader.is_encrypted:
                    raise DomainError("请上传可读取的未加密PDF资料")
                return [{"text": text, "locator": {"page": index}}
                        for index, page in enumerate(reader.pages, 1) if (text := page.extract_text() or "").strip()]
            if extension == ".docx":
                from docx import Document
                document = Document(path)
                fragments = [{"text": paragraph.text, "locator": {"paragraph": index}}
                             for index, paragraph in enumerate(document.paragraphs, 1) if paragraph.text.strip()]
                for index, table in enumerate(document.tables, 1):
                    for row_index, row in enumerate(table.rows, 1):
                        text = "\t".join(cell.text for cell in row.cells)
                        if text.strip():
                            fragments.append({"text": text, "locator": {"table": index, "row": row_index}})
                return fragments
            raise DomainError("支持导入TXT、MD、PDF和DOCX资料")
        except DomainError:
            raise
        except Exception:
            raise DomainError("资料文件读取失败，请核对文件格式和原文") from None

    @staticmethod
    def _chunk_fragments(fragments):
        if not isinstance(fragments, list) or not fragments:
            raise DomainError("资料未提取到可引用文字，请提供文本及页段定位")
        chunks = []
        for fragment in fragments:
            text, locator = fragment.get("text"), fragment.get("locator")
            if not isinstance(text, str) or not text.strip() or not isinstance(locator, dict) or not locator:
                raise DomainError("资料需要非空原文和页段定位")
            if len(text) <= 1200:
                chunks.append({"text": text, "locator": locator})
                continue
            start, index = 0, 1
            while start < len(text):
                end = min(start + 1200, len(text))
                chunks.append({"text": text[start:end], "locator": {**locator, "chunk": index,
                               "start_offset": start, "end_offset": end}})
                if end == len(text):
                    break
                start = end - 100
                index += 1
        return chunks

    @staticmethod
    def _key(entry_id, version) -> str:
        return f"{entry_id}:{version}"
