"""Prepared feature inference remains scoped to the authenticated session."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.adapters.model_runtime import ModelRuntime
from app.services.prediction.packages import ModelPackagesService


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def list_packages(self):
        return [dict(model_id="forecast", model_version="v1", kind="gas_forecast",
                     source_id="history", face_id="31002", status="ready", missing_fields=[]),
                dict(model_id="other", model_version="v1", kind="gas_attribution",
                     source_id="other-source", face_id="3411", status="ready", missing_fields=[])]

    def infer(self, model_id, version, request):
        self.calls.append((model_id, version, deepcopy(request)))
        return dict(status="completed", model_id=model_id, model_version=version, outputs=[])


@pytest.fixture
def packages():
    store = Store("sqlite+pysqlite:///:memory:")
    store.put("source", "history", dict(source_id="history", face_id="31002"))
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    context = ExecutionContext(session_id="session", source_id="history", mode="historical_replay",
                               as_of=datetime.fromisoformat("2022-02-05T19:00:00+08:00"), actor=actor)
    runtime = FakeRuntime()
    service = ModelPackagesService(store, runtime)
    yield service, runtime, context
    store.close()


def test_package_catalog_is_filtered_by_session_source_and_face(packages):
    service, _, context = packages
    assert [p["model_id"] for p in service.list_packages(context)] == ["forecast"]


def test_prepared_features_use_server_clock_and_preserve_input_digest(packages):
    service, runtime, context = packages
    features = dict(feature_names=["OT"], times=["2022-02-05T18:55:00+08:00"], values=[[.2]])
    before = deepcopy(features)
    result = service.run(context, "forecast", features)
    assert result["status"] == "completed"
    assert result["input_origin"] == "submitted_features"
    assert len(result["input_sha256"]) == 64
    assert datetime.fromisoformat(runtime.calls[0][2]["as_of"]) == context.as_of
    assert features == before
    assert service.get_run(context, result["run_id"]) == result


def test_package_outside_source_is_rejected_before_worker(packages):
    service, runtime, context = packages
    with pytest.raises(DomainError) as caught:
        service.run(context, "other", {})
    assert caught.value.status_code == 403
    assert runtime.calls == []


def test_client_cannot_override_session_clock(packages):
    service, runtime, context = packages
    with pytest.raises(DomainError):
        service.run(context, "forecast", {"as_of": "2023-01-01T00:00:00+08:00"})
    assert runtime.calls == []


def test_future_prepared_input_is_rejected_before_worker(packages):
    service, runtime, context = packages
    with pytest.raises(DomainError):
        service.run(context, "forecast", {"times": ["2022-02-05T20:00:00+08:00"], "values": [[.1]]})
    assert runtime.calls == []


def test_run_cannot_be_read_from_another_session(packages):
    service, _, context = packages
    result = service.run(context, "forecast", {})
    other = context.model_copy(update={"session_id": "other-session"})
    with pytest.raises(DomainError) as caught:
        service.get_run(other, result["run_id"])
    assert caught.value.status_code == 403


def test_empty_runtime_has_no_packages():
    assert ModelRuntime().list_packages() == []


def test_catalog_rejects_duplicate_model_versions(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps(dict(kind="gas_forecast", model_id="forecast", version="v1", artifacts={})))
    entry = dict(bundle_dir="bundle", source_id="history", face_id="31002")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(dict(models=[entry, entry])))
    with pytest.raises(ValueError):
        ModelRuntime(catalog)


def test_missing_artifact_report_does_not_require_model_dependencies(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps(dict(kind="gas_forecast", model_id="forecast", version="v1", artifacts={})))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(dict(models=[dict(bundle_dir="bundle", source_id="history", face_id="31002")])))
    runtime = ModelRuntime(catalog)
    result = runtime.infer("forecast", "v1", {})
    assert result["status"] == "not_ready"
    assert result["missing_fields"]


def test_runtime_never_accepts_paths_from_inference_request(tmp_path):
    with pytest.raises(ValueError):
        ModelRuntime().infer("unknown", "v1", {"bundle_dir": str(tmp_path)})


def test_catalog_identity_change_is_detected_even_when_artifacts_missing(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = dict(kind="gas_forecast", model_id="forecast", version="v1", artifacts={})
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(dict(models=[dict(bundle_dir="bundle", source_id="history", face_id="31002")])))
    runtime = ModelRuntime(catalog)
    manifest["version"] = "replacement"
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    assert runtime.infer("forecast", "v1", {})["status"] == "failed"


def test_authenticated_model_package_routes_preserve_session_scope(tmp_path):
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.main import create_app

    application = create_app(Settings(_env_file=None, runtime_dir=tmp_path,
                                      database_url="sqlite+pysqlite:///:memory:"))
    with TestClient(application) as client:
        services = application.state.services
        services["auth"].create_user("manager", "test-password-12345", ["safety_management"], ["31002"])
        services["model_packages"] = ModelPackagesService(services["store"], FakeRuntime())
        services["data"].register_source("history", "Synthetic", "31002")
        assert client.get("/api/model-packages", params={"session_id": "unknown"}).status_code == 401
        token = client.post("/api/auth/login", json={"username": "manager", "password": "test-password-12345"}).json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        session = client.post("/api/sessions", json={"source_id": "history", "mode": "historical_replay",
                              "start_as_of": "2022-02-05T19:00:00+08:00"}).json()["session_id"]
        listing = client.get("/api/model-packages", params={"session_id": session})
        assert listing.status_code == 200
        assert [p["model_id"] for p in listing.json()] == ["forecast"]
        run = client.post("/api/model-package-runs", json={"session_id": session, "model_id": "forecast", "features": {}})
        assert run.status_code == 200
        assert run.json()["input_origin"] == "submitted_features"
        assert client.get('/api/model-package-runs/' + run.json()["run_id"], params={"session_id": session}).json() == run.json()
