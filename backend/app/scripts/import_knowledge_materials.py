"""Repeatable local material import with explicit, version-bound approval."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

from app.db import Store
from app.schemas.common import Actor, DomainError
from app.services.knowledge.service import KnowledgeService


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def import_materials(knowledge, actor, manifest_path, approve_ids=(), review_comment=""):
    knowledge.auth.authorize(actor, "manage_data")
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise DomainError("材料清单须为本机UTF-8 JSON文件") from None
    if not isinstance(manifest, list) or not manifest or any(not isinstance(item, dict) for item in manifest):
        raise DomainError("材料清单须为非空记录数组")
    ids = [item.get("material_id") for item in manifest]
    if any(not isinstance(value, str) or not value.strip() for value in ids) or len(ids) != len(set(ids)):
        raise DomainError("材料编号须非空且唯一")
    approvals = set(approve_ids)
    if not approvals.issubset(ids) or approvals and not review_comment.strip():
        raise DomainError("批准清单须指定已有材料编号及审核说明")
    prepared = []
    for item in manifest:
        relative = item.get("relative_path")
        if not isinstance(relative, str) or not relative:
            raise DomainError("材料缺少本机相对路径")
        original = (path.parent / relative).resolve()
        if not original.is_relative_to(path.parent):
            raise DomainError("材料必须位于清单目录内", code="forbidden", status_code=403)
        try:
            if original.stat().st_size > 20 * 1024 * 1024:
                raise DomainError("单份知识材料上限20MiB")
            content = original.read_bytes()
        except OSError:
            raise DomainError("知识材料文件读取失败") from None
        if hashlib.sha256(content).hexdigest() != item.get("sha256"):
            raise DomainError("知识材料摘要与清单不一致", code="file_integrity_failed", status_code=409)
        spec = deepcopy(item.get("import_spec", {}))
        if not isinstance(spec, dict) or any(not isinstance(spec.get(key), str) or not spec[key].strip()
                for key in ("title", "source", "version")):
            raise DomainError("材料需要标题、来源、版本及适用范围")
        if set(spec) - {"title", "source", "version", "applicability", "locator"}:
            raise DomainError("材料清单只能声明资料元数据")
        spec["applicability"] = knowledge._applicability(actor, spec.get("applicability"))
        knowledge._validate_content(original.suffix.lower(), content)
        if item["material_id"] in approvals:
            if item.get("currency_status") not in ("current_user_confirmed", "current_publication_verified"):
                raise DomainError("待核材料与研究草稿保持待审")
            knowledge._require_effective(spec["applicability"], knowledge.clock())
        key = _digest([item["material_id"], item["sha256"], sorted(spec["applicability"]["face_ids"])])
        spec_digest = _digest(spec)
        previous = knowledge.store.get("knowledge_material_import", key)
        if previous is not None and previous["spec_sha256"] != spec_digest:
            raise DomainError("既有材料元数据变更须作为新版本登记", code="conflict", status_code=409)
        prepared.append((item, original, content, spec, key, spec_digest, previous))

    result = dict(imported=0, reused=0, approved=0, items=[])
    for item, original, content, spec, key, spec_digest, previous in prepared:
        if previous is not None:
            document = knowledge.get_reference(actor, previous["entry_id"], previous["version"])
            result["reused"] += 1
        else:
            document = next((entry for entry in knowledge.list_references(actor, kind="document")
                if entry.get("file_sha256") == item["sha256"] and entry["title"] == spec["title"]
                and entry["version"] == spec["version"] and entry["source"] == spec["source"]
                and entry["applicability"] == spec["applicability"]), None)
            if document is None:
                uploaded = next((entry for entry in knowledge.store.list("knowledge_file")
                    if entry["sha256"] == item["sha256"] and entry["uploaded_by"] == actor.user_id
                    and set(entry["face_ids"]) == set(spec["applicability"]["face_ids"])), None)
                if uploaded is None:
                    uploaded = knowledge.upload_file(actor, original.name, content, spec["applicability"]["face_ids"])
                document = knowledge.import_reference(actor, {**spec, "file_id": uploaded["file_id"]})
                result["imported"] += 1
            else:
                result["reused"] += 1
            knowledge.store.put("knowledge_material_import", key, {"material_id": item["material_id"],
                "sha256": item["sha256"], "spec_sha256": spec_digest,
                "entry_id": document["entry_id"], "version": document["version"]})
        if item["material_id"] in approvals and document["state"] == "pending_review":
            document = knowledge.review_reference(actor, document["entry_id"], document["version"], "approved", review_comment)
            result["approved"] += 1
        result["items"].append({key: document[key] for key in
            ("entry_id", "title", "version", "state", "fragment_count", "file_sha256")})
        result["items"][-1]["material_id"] = item["material_id"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="本机知识材料清单入库，默认全部待审")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--upload-dir", required=True)
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--face-id", action="append", required=True)
    parser.add_argument("--approve-material", action="append", default=[])
    parser.add_argument("--review-comment", default="")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    store = None
    try:
        database = Path(args.database).resolve()
        if str(database).startswith(("\\\\", "//")):
            raise DomainError("知识运行库须为本机SQLite文件")
        backup = Path(args.backup_dir).resolve() / "before-knowledge-import.sqlite3"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup == database:
            raise DomainError("备份文件须与运行库分开")
        if database.exists() and not backup.exists():
            with sqlite3.connect(database.as_uri()+"?mode=ro", uri=True) as source:
                with sqlite3.connect(backup) as target:
                    source.backup(target)
        store = Store("sqlite+pysqlite:///" + database.as_posix())
        knowledge = KnowledgeService(store, upload_dir=Path(args.upload_dir))
        actor = Actor(user_id=args.actor_id, roles=("safety_management",), face_ids=tuple(args.face_id))
        result = import_materials(knowledge, actor, args.manifest, args.approve_material, args.review_comment)
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("imported", "reused", "approved")}))
        return 0
    except DomainError as error:
        print(json.dumps(dict(error=error.code, message=error.message), ensure_ascii=False))
        return 2
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
