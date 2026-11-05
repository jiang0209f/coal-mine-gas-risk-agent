"""Collect exact local source passages as pending control measures; never approve."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET
import zipfile

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext, now_utc
from app.services.control.service import ControlService
from app.services.knowledge.service import KnowledgeService

__all__ = ["main"]


def _parser():
    parser = argparse.ArgumentParser(description="本机原文整理为待审措施，不执行任何审核")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--reference-id", action="append", default=[])
    parser.add_argument("--root-cause-docx", type=Path)
    parser.add_argument("--paragraph", type=int, action="append", default=[], help="DOCX物理段号，从1开始")
    parser.add_argument("--apply", action="store_true", help="实际写入pending；默认只读预览")
    parser.add_argument("--output", type=Path)
    return parser


def _docx(path, paragraphs):
    if not path or path.suffix.lower() != ".docx" or not path.is_file() or not paragraphs:
        raise DomainError("根因资料需要现有DOCX及明确段号")
    content = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    all_paragraphs = root.findall(".//w:p", namespace)
    fragments = []
    for number in dict.fromkeys(paragraphs):
        if number < 1 or number > len(all_paragraphs):
            raise DomainError("DOCX段号超出原文范围")
        text = "".join(node.text or "" for node in all_paragraphs[number - 1].findall(".//w:t", namespace))
        if text.strip():
            fragments.append({"text": text, "locator": {"paragraph": number}})
    return hashlib.sha256(content).hexdigest(), fragments


def _run(arguments):
    database = arguments.database.resolve()
    if not database.is_file() or not (arguments.reference_id or arguments.root_cause_docx):
        raise DomainError("需要现有运行库及明确的引用或本机原文")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        def record(kind, identifier):
            row = connection.execute("SELECT payload FROM records WHERE kind=? AND id=?", (kind, identifier)).fetchone()
            return json.loads(row[0]) if row else None

        source, user = record("source", arguments.source_id), record("users", arguments.actor_id)
        if not source or not user:
            raise DomainError("来源或本机操作账户不存在", "not_found", 404)
        actor = Actor(user_id=arguments.actor_id, roles=tuple(user["roles"]), face_ids=tuple(user["face_ids"]))
        actor.require_role("safety_management")
        actor.require_face(source["face_id"])
        references = []
        for identifier in dict.fromkeys(arguments.reference_id):
            fragment = record("reference_fragment", identifier)
            if not fragment or source["face_id"] not in fragment.get("applicability", {}).get("face_ids", []):
                raise DomainError("引用不存在或范围不符", "forbidden", 403)
            entry = record("reference", f"{fragment['entry_id']}:{fragment['version']}")
            references.append({"reference_id": identifier, "entry_id": fragment["entry_id"],
                               "title": entry.get("title") if entry else None,
                               "reference_state": entry.get("state") if entry else None,
                               "locator": fragment.get("locator"), "excerpt": fragment["text"][:400]})
    digest, root_fragments = _docx(arguments.root_cause_docx, arguments.paragraph) if arguments.root_cause_docx else (None, [])
    result = {"dry_run": not arguments.apply, "source_id": arguments.source_id, "actor_id": arguments.actor_id,
              "references": references, "root_cause_source": str(arguments.root_cause_docx) if arguments.root_cause_docx else None,
              "root_cause_sha256": digest, "root_cause_fragment_count": len(root_fragments),
              "approval_actions": 0, "created_or_existing_measures": 0, "measures": []}
    if not arguments.apply:
        return result
    store = Store("sqlite+pysqlite:///" + database.as_posix())
    try:
        knowledge = KnowledgeService(store)
        context = ExecutionContext(session_id="control-library-import", source_id=arguments.source_id,
                                   mode="historical_replay", as_of=now_utc(), actor=actor)
        identifiers = list(dict.fromkeys(arguments.reference_id))
        if root_fragments:
            source_path = str(arguments.root_cause_docx.resolve())
            version = "原文摘录待核-" + digest[:12]
            entry = next((row for row in store.list("reference") if row.get("source") == source_path and row.get("version") == version), None)
            if entry is None:
                entry = knowledge.import_reference(actor, {"title": arguments.root_cause_docx.stem + "（原文摘录，待核）",
                    "source": source_path, "version": version, "fragments": root_fragments,
                    "applicability": {"face_ids": [source["face_id"]]}})
            identifiers.extend(row["reference_id"] for row in store.list("reference_fragment")
                               if row["entry_id"] == entry["entry_id"] and row["version"] == entry["version"])
            result["root_cause_entry_id"] = entry["entry_id"]
            result["root_cause_reference_state"] = entry["state"]
        service = ControlService(store, {"knowledge": knowledge})
        measures = service.import_measures(context, identifiers)
        result["created_or_existing_measures"] = len(measures)
        result["measures"] = [{key: row.get(key) for key in (
            "measure_id", "reference_id", "entry_id", "version", "locator", "state", "reference_state",
            "cause_ids", "assigned_role", "role", "deadline", "tier")} for row in measures]
    finally:
        store.close()
    return result


def main(argv=None):
    arguments = _parser().parse_args(argv)
    result = _run(arguments)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
