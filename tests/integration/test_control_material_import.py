"""Local material collection creates pending measures without changing approvals."""
import json

from app.db import Store
from app.scripts.import_control_measures import main


def test_material_import_keeps_pending_reference_and_measure_pending(tmp_path, capsys):
    database = tmp_path / "materials.db"
    url = "sqlite+pysqlite:///" + database.as_posix()
    store = Store(url)
    store.put("source", "source", {"source_id": "source", "face_id": "face"})
    store.put("users", "operator", {"user_id": "operator", "roles": ["safety_management"], "face_ids": ["face"]})
    store.put("reference", "doc:1", {"entry_id": "doc", "version": 1, "state": "pending_review",
                                      "applicability": {"face_ids": ["face"]}})
    store.put("reference_fragment", "ref", {"reference_id": "ref", "entry_id": "doc", "version": 1,
        "kind": "document", "text": "通风路线不畅时，应停止作业并核查。", "locator": {"page": 3},
        "applicability": {"face_ids": ["face"]}})
    store.close()
    assert main(["--database", str(database), "--source-id", "source", "--actor-id", "operator",
                 "--reference-id", "ref", "--apply"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["created_or_existing_measures"] == 1
    assert result["measures"][0]["state"] == "pending"
    store = Store(url)
    assert store.get("reference", "doc:1")["state"] == "pending_review"
    assert store.list("control_measure")[0]["state"] == "pending"
    assert store.list("control_plan") == []
    store.close()
