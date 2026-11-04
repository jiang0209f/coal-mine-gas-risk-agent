"""Explicit session model selection and completed forecast -> alert -> brief linkage."""
from datetime import datetime, timedelta, timezone
from math import floor
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app
from app.schemas.common import Actor, DomainError


MOMENT = datetime(2022, 2, 5, 12, 5, tzinfo=timezone.utc)
PASSWORD = "synthetic-forecast-test-password"


class Briefs:
    def __init__(self, services):
        self.services, self.submitted, self.executed = services, [], []

    def submit(self, context, event_id, event_revision=None, schedule=None):
        snapshot = self.services["alerts"].get_event_snapshot(context, event_id, event_revision)
        assert snapshot["forecast_run_id"]
        self.submitted.append(snapshot)
        task_id = "brief-" + str(len(self.submitted))
        schedule(lambda: self.executed.append(task_id))
        return dict(event_id=event_id, event_revision=event_revision, task_id=task_id, status="queued")


@pytest.fixture
def api(tmp_path):
    app = create_app(Settings(_env_file=None, runtime_dir=tmp_path,
        database_url="sqlite+pysqlite:///:memory:", model_catalog_path=None,
        deepseek_api_key=None, event_brief_allowed_source_ids=[], ingestion_worker_enabled=False))
    with TestClient(app) as client:
        services = app.state.services
        user = services["auth"].create_user("forecast-test", PASSWORD, ["safety_management"], ["31002"])
        services["auth"].create_user("other-test", PASSWORD, ["safety_management"], ["32001"])
        actor = Actor(user_id=user["user_id"], roles=("safety_management",), face_ids=("31002",))
        for source_id, face in (("source", "31002"), ("other-source", "32001")):
            services["data"].register_source(source_id, source_id, face)
            services["data"].register_point(source_id + "-t0", source_id, "T0", "T0", "methane", "%", unit_status="confirmed")
        services["data"].register_point("source-t1", "source", "T1", "T1", "methane", "%", unit_status="confirmed")
        services["data"].ingest_readings(actor, "source", [{"reading_id": str(i), "point_id": "source-t0",
            "event_time": (MOMENT - timedelta(minutes=5-i)).isoformat(), "raw_value": ".40"} for i in range(5)],
            received_at=MOMENT-timedelta(seconds=1))
        common = dict(version="v1", face_id="31002", source_id="source", point_ids=["source-t0"],
            target_quantity="methane", unit="%", aggregation_seconds=300, input_window=1,
            horizon_seconds=3600, allowed_lineage=["raw"])
        services["prediction"].register_model(actor, dict(common, model_id="baseline", adapter_kind="persistence", default_eligible=True))
        def professional(model, rows, horizon):
            output = services["prediction"]._persistence(model, rows, horizon)
            output[0]["prediction_stage"] = "residual_corrected"
            for interval in output[0]["interval_values"]:
                interval["value"] = .9
            return output
        services["prediction"].register_adapter("synthetic-professional", professional)
        services["prediction"].register_model(actor, dict(common, model_id="professional", adapter_kind="synthetic-professional", default_eligible=False))
        capture = services["event_briefs"] = Briefs(services)
        session = services["sessions"].create_session(actor, "historical_replay", "source", MOMENT)
        token = client.post("/api/auth/login", json={"username":"forecast-test", "password":PASSWORD}).json()["access_token"]
        client.headers["Authorization"] = "Bearer " + token
        yield client, services, actor, session, capture


def configure(client, sid, **changes):
    return client.post(f"/api/sessions/{sid}/prediction-models", json={"point_id":"source-t0", "model_id":"professional", "version":"v1", **changes})


def test_manual_professional_forecast_creates_linked_alert_and_only_one_grade_brief(api):
    client, services, actor, session, capture = api
    body = dict(session_id=session["session_id"], point_id="source-t0", model_id="professional", version="v1")
    first = client.post("/api/forecasts", json=body)
    assert first.status_code == 200
    run = first.json()
    assert run["status"] == "completed" and run["assessment"]["alert_level"] == "orange"
    assert len(run["event_refs"]) == len(run["briefs"]) == len(capture.submitted) == len(capture.executed) == 1
    event = capture.submitted[0]
    assert event["forecast_run_id"] == run["run_id"]
    assert event["event_type"] == "forecast_exceedance"
    again = client.post("/api/forecasts", json=body)
    assert again.status_code == 200 and again.json()["assessment"]["alert_level"] == "orange"
    assert len(capture.submitted) == 1
    assert services["prediction"].get_forecast(services["sessions"].get_context(actor,session["session_id"]),run["run_id"])["outputs"] == run["outputs"]


def test_default_and_manual_baseline_stay_baseline_without_forecast_alert(api):
    client, services, actor, session, capture = api
    body = dict(session_id=session["session_id"], point_id="source-t0")
    response = client.post("/api/forecasts", json=body)
    assert response.status_code == 200
    run = response.json()
    assert run["model_id"] == "baseline" and run["outputs"][0]["prediction_stage"] == "baseline_persistence"
    assert run["assessment"]["forecast_status"] == "model_not_ready" and run["event_refs"] == [] and run["briefs"] == []
    assert capture.submitted == []
    model = services["prediction"].get_model(services["sessions"].get_context(actor,session["session_id"]),"professional","v1")
    assert model["default_eligible"] is False


def test_selected_session_model_is_persisted_audited_and_invalidates_same_tick_cache(api):
    client, services, actor, session, capture = api
    sid = session["session_id"]
    endpoint = f"/api/sessions/{sid}/tick"
    body = {"target_as_of":MOMENT.isoformat()}
    assert client.post(endpoint,json=body).json()["forecasts"][0]["model_id"] == "baseline"
    response = configure(client,sid)
    assert response.status_code == 200, response.text
    binding = {"model_id":"professional", "version":"v1"}
    assert response.json()["prediction_models"] == {"source-t0":binding}
    assert client.get(f"/api/sessions/{sid}").json()["prediction_models"]["source-t0"] == binding
    audit = sorted(services["store"].list("session_action"),key=lambda row:row["sequence"])[-1]
    assert audit["action_type"] == "set_prediction_model"
    assert audit["before"]["prediction_models"] == {} and audit["after"]["prediction_models"] == {"source-t0":binding}
    result = client.post(endpoint,json=body)
    assert result.status_code == 200, result.text
    assert result.json()["forecasts"][0]["model_id"] == "professional"
    assert not result.json().get("reused",False) and len(capture.submitted) == 1
    repeated = client.post(endpoint,json=body).json()
    assert repeated["reused"] and len(capture.submitted) == 1


def test_selected_missing_model_never_falls_back_to_default(api):
    client, services, _, session, capture = api
    sid = session["session_id"]
    assert configure(client,sid).status_code == 200
    services["store"].delete("model","professional:v1")
    response = client.post(f"/api/sessions/{sid}/tick",json={"target_as_of":MOMENT.isoformat()})
    assert response.status_code == 200
    run = response.json()["forecasts"][0]
    assert run["model_id"] == "professional" and run["model_version"] == "v1"
    assert run["status"] == "not_ready" and "model_version" in run["missing_fields"] and run["outputs"] == []
    assert capture.submitted == []


@pytest.mark.parametrize("changes,status", [({"point_id":"other-source-t0"},403),
    ({"point_id":"source-t1"},403), ({"model_id":"missing"},404), ({"version":"missing"},404)])
def test_selection_validates_point_model_and_exact_version(api,changes,status):
    client, services, actor, session, _ = api
    response = configure(client,session["session_id"],**changes)
    assert response.status_code == status
    assert services["sessions"].get_session(actor,session["session_id"]).get("prediction_models",{}) == {}


def test_selection_rejects_missing_version_unknown_fields_cross_face_and_system_actor(api):
    client, services, actor, session, _ = api
    sid = session["session_id"]
    endpoint = f"/api/sessions/{sid}/prediction-models"
    assert client.post(endpoint,json={"point_id":"source-t0","model_id":"professional"}).status_code == 422
    assert configure(client,sid,source_id="other-source").status_code == 422
    token = client.post("/api/auth/login",json={"username":"other-test","password":PASSWORD}).json()["access_token"]
    client.headers["Authorization"] = "Bearer " + token
    assert configure(client,sid).status_code == 403
    with pytest.raises(DomainError) as error:
        services["sessions"].set_prediction_model(actor.model_copy(update={"is_system":True}),sid,"source-t0","professional","v1")
    assert error.value.status_code == 403


def test_ended_session_cannot_change_selection(api):
    client, services, actor, session, _ = api
    sid = session["session_id"]
    services["sessions"].apply_action(actor,sid,"stop")
    assert configure(client,sid).status_code == 409


def test_selected_trained_model_reuses_completed_cutoff_but_refreshes_observed_checks(api):
    client, services, _, session, _ = api
    model = services["store"].get("model","professional:v1")
    model["adapter_kind"] = "fedformer_lgbm"
    services["store"].put("model","professional:v1",model)
    calls = []
    def trained_forecast(context,point_id,model,horizon):
        calls.append(context.as_of)
        cutoff = datetime.fromtimestamp(floor(context.as_of.timestamp()/300)*300,timezone.utc)
        outputs = services["prediction"]._persistence(model,[{"bucket_end":cutoff.isoformat(),"value":.9}],horizon)
        outputs[0]["prediction_stage"] = "residual_corrected"
        return dict(status="completed",outputs=outputs,input_refs=[],input_origin="server_observations",feature_pipeline_version="synthetic-v1")
    services["prediction"].professional_pipeline = SimpleNamespace(forecast=trained_forecast)
    sid = session["session_id"]
    assert configure(client,sid).status_code == 200
    path = f"/api/sessions/{sid}/tick"
    first = client.post(path,json={"target_as_of":MOMENT.isoformat()}).json()
    second = client.post(path,json={"target_as_of":(MOMENT+timedelta(seconds=30)).isoformat()}).json()
    assert len(calls) == 1
    assert second["forecasts"][0]["run_id"] == first["forecasts"][0]["run_id"]
    assert second["forecasts"][0]["reused"] is True
    assert second["assessments"][0]["as_of"] != first["assessments"][0]["as_of"]
    third = client.post(path,json={"target_as_of":(MOMENT+timedelta(minutes=5)).isoformat()}).json()
    assert len(calls) == 2 and third["forecasts"][0]["run_id"] != first["forecasts"][0]["run_id"]


def test_linked_replay_copies_explicit_model_choices_and_keeps_them_independent(api):
    client, services, actor, parent, _ = api
    assert configure(client,parent["session_id"]).status_code == 200
    response = client.post("/api/sessions",json={"mode":"historical_replay","source_id":"source",
        "start_as_of":(MOMENT-timedelta(minutes=10)).isoformat(),"parent_session_id":parent["session_id"]})
    assert response.status_code == 200
    child = response.json()
    assert child["prediction_models"] == {"source-t0":{"model_id":"professional","version":"v1"}}
    assert configure(client,child["session_id"],model_id="baseline").status_code == 200
    assert services["sessions"].get_session(actor,parent["session_id"])["prediction_models"]["source-t0"]["model_id"] == "professional"
    assert services["sessions"].get_session(actor,child["session_id"])["prediction_models"]["source-t0"]["model_id"] == "baseline"
    assert services["store"].list("forecast_run") == []
