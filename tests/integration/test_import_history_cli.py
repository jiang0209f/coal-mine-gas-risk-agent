import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from openpyxl import Workbook
import pytest

from app.db import Store
from app.scripts.import_history import main


@pytest.fixture
def configuration(tmp_path):
    return SimpleNamespace(runtime_dir=tmp_path / "runtime", database_url="sqlite:///" + (tmp_path / "runtime" / "db.sqlite").as_posix())


def arguments(path, **overrides):
    options = {"path": str(path), "source-id": "3411-looking-source", "source-name": "Explicit source", "face-id": "31002", "point-id": "31002:T0", "source-tag": "RAW.TAG", "point-name": "T0 Methane", "start": "2026-10-01T08:00:00", "end": "2026-10-01T08:02:00"}
    options.update(overrides)
    return [part for key, value in options.items() for part in (f"--{key}", value)]


def test_cli_imports_only_requested_range_and_outputs_quality(configuration, tmp_path, capsys):
    file = tmp_path / "small.csv"
    file.write_text("STime,SValue\n2026-10-01 07:59:59,0.2\n2026-10-01 08:00:00,0.8\n2026-10-01 08:01:00,bad\n2026-10-01 08:02:00,0.4\n", encoding="utf-8")
    before = file.read_bytes()
    before_mtime = file.stat().st_mtime_ns
    assert main(arguments(file), settings=configuration) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["report"]["imported_rows"] == 2
    assert output["report"]["skipped_rows"] == 2
    assert output["source"]["face_id"] == "31002"
    assert output["query_quality"][0]["total"] == 2
    assert output["query_quality"][0]["quality"]["valid_rows"] == 1
    assert output["query_quality"][0]["quality"]["invalid_rows"] == 1
    assert output["report"]["start_at"] == "2026-10-01T00:00:00.000000+00:00"
    assert file.read_bytes() == before
    assert file.stat().st_mtime_ns == before_mtime
    assert output["file"]["sha256"] == hashlib.sha256(before).hexdigest()


def test_cli_supports_mapping_json(configuration, tmp_path, capsys):
    file = tmp_path / "custom.csv"
    file.write_text("when,CH4\n2026-10-01 08:00:00,0.6\n", encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"time_column": "when", "value_columns": {"CH4": "31002:T0"}, "lineage": "raw"}), encoding="utf-8")
    assert main(arguments(file, **{"mapping-file": str(mapping)}), settings=configuration) == 0
    assert json.loads(capsys.readouterr().out)["report"]["imported_rows"] == 1


def test_cli_selects_requested_xlsx_sheet(configuration, tmp_path, capsys):
    file = tmp_path / "small.xlsx"
    workbook = Workbook()
    workbook.active.title = "Ignore"
    workbook.active.append(["unrelated"])
    sheet = workbook.create_sheet("Raw")
    sheet.append(["STime", "SValue"])
    sheet.append(["2026-10-01 08:00:00", 0.7])
    workbook.save(file)
    workbook.close()
    assert main(arguments(file, sheet="Raw"), settings=configuration) == 0
    assert json.loads(capsys.readouterr().out)["report"]["imported_rows"] == 1


def test_cli_rejects_reversed_range_before_import(configuration, tmp_path, capsys):
    file = tmp_path / "small.csv"
    file.write_text("STime,SValue\n", encoding="utf-8")
    assert main(arguments(file, start="2026-10-01T09:00:00"), settings=configuration) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "invalid_request"


def test_cli_does_not_infer_face_from_source_prefix(configuration, tmp_path, capsys):
    file = tmp_path / "small.csv"
    file.write_text("STime,SValue\n2026-10-01 08:00:00,0.4\n", encoding="utf-8")
    assert main(arguments(file), settings=configuration) == 0
    capsys.readouterr()
    store = Store(configuration.database_url)
    try:
        assert store.get("source", "3411-looking-source")["face_id"] == "31002"
    finally:
        store.close()


def test_cli_reports_malformed_mapping_as_json(configuration, tmp_path, capsys):
    file = tmp_path / "small.csv"
    file.write_text("STime,SValue\n", encoding="utf-8")
    mapping = tmp_path / "broken.json"
    mapping.write_text("{invalid", encoding="utf-8")
    assert main(arguments(file, **{"mapping-file": str(mapping)}), settings=configuration) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_cli_defaults_database_to_runtime_app_db(configuration, tmp_path, capsys):
    configuration.database_url = None
    file = tmp_path / "small.csv"
    file.write_text("STime,SValue\n2026-10-01 08:00:00,0.4\n", encoding="utf-8")
    assert main(arguments(file), settings=configuration) == 0
    capsys.readouterr()
    assert (configuration.runtime_dir / "app.db").is_file()


def test_cli_uses_runtime_dir_environment_without_dotenv(tmp_path, capsys, monkeypatch):
    runtime = tmp_path / "environment-runtime"
    monkeypatch.setenv("RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    file = tmp_path / "small.csv"
    file.write_text("STime,SValue\n2026-10-01 08:00:00,0.4\n", encoding="utf-8")
    assert main(arguments(file)) == 0
    capsys.readouterr()
    assert (runtime / "app.db").is_file()


def test_cli_registers_explicit_metadata_for_each_mapped_point(configuration, tmp_path, capsys):
    file = tmp_path / "two-points.csv"
    file.write_text("STime,A,B\n2026-10-01 08:00:00,0.4,0.9\n", encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "time_column": "STime", "value_columns": {"A": "31002:T0", "B": "31002:T2"},
        "lineage": "raw", "points": [{"point_id": "31002:T2", "source_tag": "TAG.T2",
            "name": "T2 Methane", "quantity": "methane", "unit": "%"}],
    }), encoding="utf-8")
    assert main(arguments(file, **{"mapping-file": str(mapping)}), settings=configuration) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["report"]["imported_rows"] == 2
    assert {packet["point_id"] for packet in result["query_quality"]} == {"31002:T0", "31002:T2"}
    store = Store(configuration.database_url)
    try:
        assert store.get("point", "31002:T2")["source_tag"] == "TAG.T2"
        assert store.get("point", "31002:T2")["face_id"] == "31002"
    finally:
        store.close()


def test_cli_multi_point_mapping_reports_missing_metadata(configuration, tmp_path, capsys):
    file = tmp_path / "two-points.csv"
    file.write_text("STime,A,B\n2026-10-01 08:00:00,0.4,0.9\n", encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"value_columns": {"A": "31002:T0", "B": "31002:T2"}}), encoding="utf-8")
    assert main(arguments(file, **{"mapping-file": str(mapping)}), settings=configuration) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "invalid_request"
    assert result["missing_fields"] == ["points[31002:T2]"]


def test_advanced_primary_point_uses_explicit_quantity_and_unit(configuration, tmp_path, capsys):
    file = tmp_path / "speed.csv"
    file.write_text("STime,Speed\n2026-10-01 08:00:00,-1.2\n", encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"value_columns": {"Speed": "31002:T0"}, "points": [{
        "point_id": "31002:T0", "source_tag": "CMJ.SPEED", "name": "Shearer speed",
        "quantity": "shearer_speed", "unit": "m/min", "lineage": "raw",
    }]}), encoding="utf-8")
    assert main(arguments(file, **{"mapping-file": str(mapping)}), settings=configuration) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["query_quality"][0]["unit"] == "m/min"
    assert result["query_quality"][0]["quality"]["valid_rows"] == 1
