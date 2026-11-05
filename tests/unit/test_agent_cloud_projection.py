"""Offline boundary tests for real-task API summaries."""

from copy import deepcopy
from datetime import datetime, timedelta
import json
from types import SimpleNamespace

import pytest

from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.agent.cloud_projection import RealDataProjection


def call(name, arguments):
    return {"id": "call-1", "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


@pytest.fixture
def parts():
    context = ExecutionContext(session_id="PRIVATE_SESSION", source_id="PRIVATE_SOURCE",
        mode="historical_replay", as_of=datetime.fromisoformat("2026-11-05T12:00:00+08:00"),
        actor=Actor(user_id="PRIVATE_USER", roles=("safety_management",), face_ids=("PRIVATE_FACE",)))
    event = dict(event_id="private-event", point_id="private-point", session_id=context.session_id,
        source_id=context.source_id, as_of=context.as_of.isoformat(), event_type="observed_exceedance", alert_level="red")
    run = dict(run_id="private-run", point_id="private-point", session_id=context.session_id,
        source_id=context.source_id, as_of=context.as_of.isoformat(), status="completed")
    public = {"public-ref": dict(reference_id="public-ref", title="公开法规", version="2026",
        text="第一条 核查甲烷监测。", locator={"page": 2, "clause": "第一条", "path": "PRIVATE_PATH"},
        official_url="https://www.gov.cn/public-rule", disclosure="public_normative", effective=True,
        private_metadata="PRIVATE_METADATA")}
    checked = []

    def lookup(value, expected, result):
        if value != expected:
            raise DomainError("PRIVATE_ERROR", code="not_found")
        return deepcopy(result)

    def approved(ctx, reference_id):
        checked.append(reference_id)
        return lookup(reference_id, "public-ref", public["public-ref"])

    services = {
        "alerts": SimpleNamespace(get_event_snapshot=lambda context, event_id: lookup(event_id, "private-event", event)),
        "prediction": SimpleNamespace(get_forecast=lambda context, run_id: lookup(run_id, "private-run", run)),
        "knowledge": SimpleNamespace(approved_public_fragment=approved),
    }
    task = {"selected_point_id": "private-point"}
    points = [dict(point_id="private-point", source_id=context.source_id, unit="%", quantity="methane"),
              dict(point_id="foreign-point", source_id="other", unit="PRIVATE_UNIT")]
    models = [dict(model_id="private-model", source_id=context.source_id, file_path="PRIVATE_PATH",
                   model_type="professional", deployment_stage="experimental")]
    projection = RealDataProjection(context, task, services, points, models)
    return projection, context, task, public, checked


def test_initial_messages_contain_only_fixed_intents_and_scoped_aliases(parts):
    projection, _, task, _, _ = parts
    messages = projection.messages("PRIVATE_MINE 人员PRIVATE_PERSON 查询private-point最近一小时预测private-model，event_id=private-event run_id=private-run reference_id=public-ref，foreign-point")
    encoded = json.dumps(messages, ensure_ascii=False)
    for secret in ("PRIVATE_MINE", "PRIVATE_PERSON", "private-point", "private-model", "private-event", "private-run", "public-ref", "foreign-point", "PRIVATE_SOURCE", "PRIVATE_SESSION", "PRIVATE_PATH"):
        assert secret not in encoded
    assert "query_series" in encoded and "run_prediction" in encoded
    assert "P1" in encoded and "E1" in encoded and "R1" in encoded and "M1" in encoded and "REF1" in encoded
    assert "2000-01-02T00:00:00+00:00" in encoded
    assert task["_cloud_bindings"]
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"


def test_decode_call_restores_aliases_and_relative_clock_without_mutating_input(parts):
    projection, context, *_ = parts
    projection.messages("查询private-point")
    original = call("query_series", {"point_id": "P1", "start_at": "2000-01-01T23:00:00+00:00", "end_at": "2000-01-02T00:00:00Z"})
    saved = deepcopy(original)
    decoded = json.loads(projection.decode_call(original)["function"]["arguments"])
    assert decoded["point_id"] == "private-point"
    assert datetime.fromisoformat(decoded["start_at"]) == context.as_of - timedelta(hours=1)
    assert datetime.fromisoformat(decoded["end_at"]) == context.as_of
    assert original == saved


def test_scoped_catalogs_are_available_as_aliases_without_raw_metadata(parts):
    projection = parts[0]
    encoded = json.dumps(projection.messages("查询当前状态"), ensure_ascii=False)
    assert '"model_id": "M1"' in encoded.replace('\\"', '"')
    assert '"point_id": "P1"' in encoded.replace('\\"', '"')
    assert "experimental" in encoded and "methane" in encoded
    assert "private-model" not in encoded and "private-point" not in encoded
    assert "foreign-point" not in encoded and "PRIVATE_PATH" not in encoded


@pytest.mark.parametrize("rules", [[], [{"threshold": 0.65}], "PRIVATE_RULE", None])
def test_unknown_rule_shapes_are_dropped(parts, rules):
    output = parts[0].project("get_event_evidence", {"rules": rules})
    assert output["rules"] == {}


@pytest.mark.parametrize("question,key,seconds", [
    ("查询最近五分钟监测趋势", "lookback_seconds", 300),
    ("查询过去两小时监测数据", "lookback_seconds", 7200),
    ("预测未来三十分钟", "horizon_seconds", 1800),
    ("预测未来15分钟", "horizon_seconds", 900),
])
def test_relative_window_survives_without_original_question(parts, question, key, seconds):
    intent = json.loads(parts[0].messages(question)[-1]["content"])
    assert intent[key] == seconds
    assert question not in json.dumps(intent, ensure_ascii=False)


@pytest.mark.parametrize("arguments", [
    {"point_id": "private-point"}, {"point_id": "P999"}, {"point_id": "E1"},
    {"point_id": "P1", "start_at": "2000-01-01T23:00:00"},
    {"point_id": "P1", "start_at": "2026-11-05T00:00:00+00:00"},
])
def test_unknown_ids_raw_ids_wrong_families_and_nonvirtual_times_fail(parts, arguments):
    projection = parts[0]
    projection.messages("查询private-point event_id=private-event")
    with pytest.raises(DomainError) as error:
        projection.decode_call(call("query_series", arguments))
    assert error.value.code == "invalid_tool_arguments"


def test_series_is_aggregated_and_drops_raw_rows_and_untrusted_strings(parts):
    projection = parts[0]
    projected = projection.project("query_series", {"status": "success", "data": {
        "point_id": "private-point", "unit": "PRIVATE_UNIT", "total": 4,
        "rows": [{"value": value, "event_time": "PRIVATE_TIME", "source_file_id": "PRIVATE_FILE"}
                 for value in (0.2, 0.6, None, float("nan"))], "lineage": "PRIVATE_PATH"}})
    encoded = json.dumps(projected, allow_nan=False)
    assert "PRIVATE" not in encoded and "rows" not in encoded
    summary = projected["data"]["summary"]
    assert summary == {"count": 4, "valid_count": 2, "minimum": 0.2, "maximum": 0.6, "last": 0.6, "mean": 0.4}
    assert "unit" not in projected["data"]


def test_event_list_is_bounded_and_registers_returned_ids(parts):
    projection, context, *_ = parts
    events = [dict(event_id=f"event-{i}", point_id="private-point", source_id=context.source_id,
                   session_id=context.session_id, event_type="observed_exceedance", alert_level="red",
                   start_at=context.as_of.isoformat(), description="PRIVATE_NOTE") for i in range(20)]
    result = projection.project("query_events", events)
    assert len(result) == 12 and result[0]["event_id"] == "E1"
    assert "PRIVATE_NOTE" not in json.dumps(result)
    assert json.loads(projection.decode_call(call("get_event_evidence", {"event_id": "E1"}))["function"]["arguments"])["event_id"] == "event-0"


def test_forecast_and_attribution_preserve_output_meaning_and_finite_values(parts):
    projection, context, *_ = parts
    result = projection.project("get_event_evidence", {
        "event_snapshot": {"event_id": "private-event", "point_id": "private-point", "event_type": "observed_exceedance", "alert_level": "red"},
        "observation_fragments": [{"value": 0.4, "id": "PRIVATE_OBSERVATION"}],
        "candidate_attribution": {"status": "completed", "probability_semantics": "cluster_membership", "causal_status": "candidate_attribution",
            "candidate_causes": ["通风系统异常", "PRIVATE_CAUSE"], "cluster_probabilities": [{"cluster_id": 0, "probability": 0.8, "path": "PRIVATE_PATH"}]},
        "forecast_run": {"run_id": "private-run", "as_of": context.as_of.isoformat(), "status": "completed",
            "outputs": [{"prediction_stage": "decision_score", "quantity_semantics": "decision_score", "unit": "dimensionless",
                         "interval_values": [{"start_at": context.as_of.isoformat(), "value": 0.6, "note": "PRIVATE_NOTE"} for _ in range(20)]}]}})
    assert result["attribution"]["probability_semantics"] == "cluster_membership"
    assert result["attribution"]["candidate_causes"] == ["通风系统异常"]
    assert result["attribution"]["cluster_probabilities"] == [{"cluster_id": 0, "probability": 0.8}]
    output = result["forecast"]["outputs"][0]
    assert output["quantity_semantics"] == "decision_score" and output["unit"] == "dimensionless"
    assert len(output["interval_values"]) == 12
    assert "PRIVATE" not in json.dumps(result)


def test_public_fragments_are_revalidated_and_private_body_never_leaves(parts):
    projection, _, _, public, checked = parts
    projected = projection.project("search_documents", [
        {"reference_id": "private-ref", "text": "PRIVATE_RULE", "title": "PRIVATE_MINE"},
        {"reference_id": "public-ref", "text": "FORGED_BODY", "locator": {"path": "PRIVATE_PATH"}}])
    assert projected[0] == {"reference_id": "REF1", "availability": "local_only"}
    assert len(projected) == 2 and projected[1]["reference_id"] == "REF2"
    assert projected[1]["text"] == "第一条 核查甲烷监测。"
    assert "PRIVATE" not in json.dumps(projected) and "FORGED" not in json.dumps(projected)
    before = len(checked)
    projection.validate_public_references()
    assert len(checked) > before
    public["public-ref"]["text"] = "已变更条文"
    with pytest.raises(DomainError) as error:
        projection.validate_public_references()
    assert error.value.code == "public_reference_unavailable"


def test_local_reference_can_be_used_for_a_local_advice_draft(parts):
    projection = parts[0]
    projection.project("get_event_evidence", {"event_snapshot": {"event_id": "private-event"}})
    projection.project("search_documents", [{"reference_id": "private-ref", "text": "PRIVATE_RULE"}])
    args = json.loads(projection.decode_call(call("draft_advice", {"event_id": "E1", "reference_ids": ["REF1"]}))["function"]["arguments"])
    assert args == {"event_id": "private-event", "reference_ids": ["private-ref"]}
    result = projection.project("draft_advice", {"state": "draft", "reference_ids": ["private-ref"], "items": [{"text": "PRIVATE_RULE"}]})
    assert result["reference_ids"] == ["REF1"]
    assert result["references"] == [{"reference_id": "REF1", "availability": "local_only"}]
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("status", ["team-draft", "approved"])
def test_rule_approval_status_is_kept_with_thresholds(parts, status):
    result = parts[0].project("get_event_evidence", {"rules": {"status": status, "observed_red_threshold": 1.0}})
    assert result["rules"] == {"status": status, "observed_red_threshold": 1.0}


@pytest.mark.parametrize("stage", ["base_prediction", "residual", "residual_corrected", "decision_score"])
def test_real_forecast_stages_and_fixed_dimensionless_unit_are_retained(parts, stage):
    result = parts[0].project("get_forecast", {"outputs": [{"prediction_stage": stage,
        "quantity_semantics": "decision_score", "unit": "1", "interval_values": [{"value": 0.7}]}]})
    assert result["outputs"][0]["prediction_stage"] == stage
    assert result["outputs"][0]["unit"] == "1"


@pytest.mark.parametrize("unit", ["%CH4", "%CH4（推定）"])
def test_existing_methane_units_are_kept(parts, unit):
    result = parts[0].project("query_series", {"unit": unit, "rows": [{"value": 0.4}]})
    assert result["unit"] == unit


@pytest.mark.parametrize("status", ["none", "resolved", "superseded", "possible_conflict", "open"])
def test_public_conflict_status_is_retained_without_private_conflict_metadata(parts, status):
    projection, _, _, public, _ = parts
    public["public-ref"].update(conflict_status=status, conflicts=[{"note": "PRIVATE_CONFLICT"}])
    result = projection.project("search_documents", [{"reference_id": "public-ref"}])
    assert result[0]["conflict_status"] == status
    assert "PRIVATE_CONFLICT" not in json.dumps(result)


def test_system_explains_relative_times_and_conflict_meaning(parts):
    system = parts[0].messages("检索预警条文")[0]["content"]
    assert "相对当前时刻" in system and "虚拟日期只用于工具参数" in system
    assert "open" in system and "possible_conflict" in system and "superseded" in system


def test_case_and_error_projection_never_include_action_or_exception_text(parts):
    projection = parts[0]
    case = projection.project("get_case", {"case": {"phase": "confirmed", "operation_context": "PRIVATE_OPERATION"},
        "pending_items": ["review_advice", "PRIVATE_TASK"], "actions": [{"body": "PRIVATE_PERSON"}]})
    assert case == {"case": {"phase": "confirmed"}, "pending_items": ["review_advice"], "action_count": 1}
    assert projection.project("get_case", {"status": "error", "error_code": "forbidden", "message": "PRIVATE_ERROR"}) == {"status": "error", "error_code": "forbidden"}
    assert projection.project("unknown", {"text": "PRIVATE_UNKNOWN"}) == {}


def test_restore_text_replaces_only_known_standalone_aliases_and_survives_reload(parts):
    projection, context, task, *_ = parts
    projection.messages("查询private-point event_id=private-event")
    assert projection.restore_text("P1，[E1] AP1X P10 E999") == "private-point，[private-event] AP1X P10 E999"
    loaded = RealDataProjection(context, deepcopy(task), projection.services, projection.points, projection.models)
    assert loaded.restore_text("point_id=P1") == "point_id=private-point"
