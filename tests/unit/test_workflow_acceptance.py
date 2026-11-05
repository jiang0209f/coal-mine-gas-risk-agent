"""Competition workflow acceptance uses actual outputs and event/item denominators."""
from copy import deepcopy

from app.services.evaluation.workflow_acceptance import build_acceptance


def task(identifier="task"):
    return {"task_id": identifier, "state": "failed", "expected_tools": ["query_events"],
            "expected_outputs": {"events.0.event_id": "event"},
            "tool_calls": [{"name": "query_events", "status": "success"}],
            "actual_output": {"events": [{"event_id": "event"}]}}


def event(identifier="event"):
    return {"event_id": identifier, "event_revision": 1, "point_id": "point",
            "start_at": "2026-11-05T10:00:00+00:00", "end_at": "2026-11-05T10:05:00+00:00",
            "observation_refs": [{"observation_id": "observation", "source_file_id": "file", "row_number": 1}],
            "model_or_rule_refs": [{"kind": "rule", "rule_version": "team-draft-v1"}]}


def item():
    return {"measure_id": "measure", "reference_id": "reference", "measure_state": "approved",
            "reference_state": "approved", "locator": {"page": 4}, "applicability_check": "passed",
            "text": "核查通风设施。", "source_text": "通风异常时，核查通风设施。发现问题及时报告。"}


def test_task_completion_uses_required_calls_and_output_assertions_not_state():
    good = task("good")
    wrong = task("wrong")
    wrong.update(state="completed", actual_output={"events": []})
    missing_tool = task("missing-tool")
    missing_tool.update(state="completed", tool_calls=[])
    result = build_acceptance([good, wrong, missing_tool], [], [], [])
    metric = result["tool_collaboration"]
    assert metric["numerator"] == 1 and metric["denominator"] == 3
    assert metric["rate"] == 1 / 3
    assert not metric["task_count_requirement_met"]
    assert metric["per_task"][0]["task_id"] == "good" and metric["per_task"][0]["complete"]


def test_twenty_one_unique_tasks_are_required_and_duplicate_ids_do_not_inflate_count():
    unique = [task(f"task-{index}") for index in range(21)]
    assert build_acceptance(unique, [], [], [])["tool_collaboration"]["task_count_requirement_met"]
    duplicate = [task("same")] * 21
    metric = build_acceptance(duplicate, [], [], [])["tool_collaboration"]
    assert metric["independent_task_count"] == 1
    assert not metric["task_count_requirement_met"]


def test_evidence_is_four_classes_per_event_instead_of_any_task_reference():
    incomplete = event("missing")
    incomplete.pop("model_or_rule_refs")
    result = build_acceptance([], [event(), incomplete], [], [])
    metric = result["evidence_completeness"]
    assert (metric["numerator"], metric["denominator"], metric["rate"]) == (1, 2, .5)
    assert not metric["requirement_met"]
    assert metric["per_event"][1]["missing_fields"] == ["model_or_rule_refs"]


def test_measure_coverage_requires_current_approval_locator_scope_and_content_support():
    valid = item()
    wrong = item()
    wrong.update(measure_id="wrong", text="调整无关设备。")
    metric = build_acceptance([], [], [], [{"plan_id": "plan", "items": [valid, wrong]}])["measure_basis_coverage"]
    assert metric["numerator"] == 1 and metric["denominator"] == 2
    assert metric["per_measure"][0]["consistency_method"] == "exact_excerpt"
    assert not metric["per_measure"][1]["complete"]
    assert not metric["requirement_met"]
    valid["reference_current_state"] = "superseded"
    assert build_acceptance([], [], [], [{"items": [valid]}])["measure_basis_coverage"]["numerator"] == 0


def test_report_evidence_is_bound_to_event_and_inputs_are_unchanged():
    original = event()
    original.pop("model_or_rule_refs")
    original.pop("observation_refs")
    report = {"report_id": "report", "event_id": "event", "event_revision": 1,
              "observation_refs": event()["observation_refs"], "model_or_rule_refs": event()["model_or_rule_refs"]}
    before = deepcopy([original, report])
    metric = build_acceptance([], [original], [report], [])["evidence_completeness"]
    assert metric["numerator"] == 1
    report["event_revision"] = 2
    assert build_acceptance([], [original], [report], [])["evidence_completeness"]["numerator"] == 0
    report["event_revision"] = 1
    assert [original, report] == before


def test_zero_denominators_remain_unassessed():
    result = build_acceptance([], [], [], [])
    for key in ("tool_collaboration", "evidence_completeness", "measure_basis_coverage"):
        assert result[key]["rate"] is None
        assert result[key]["denominator"] == 0
