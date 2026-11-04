"""条文分歧只按登记及同条件结构化数值提示，保留原文和定位。"""
from copy import deepcopy

import pytest

from app.services.knowledge.conflicts import annotate_conflicts


def fragment(reference_id, *, entry_id="doc", version="v1", page=1, text="合成条文", **extra):
    return {"reference_id": reference_id, "entry_id": entry_id, "version": version,
            "locator": {"page": page}, "text": text, **extra}


def conflict(**extra):
    return {"conflict_id": "conflict-1", "reference_ids": ["a", "b"], "topic": "标校周期",
            "applicability_condition": "同一甲烷传感器型号及使用场景", "reason": "已核对原文周期不一致",
            "status": "open", **extra}


def citation(ref):
    return {key: ref[key] for key in ("reference_id", "entry_id", "version", "locator")}


def test_dates_and_authority_do_not_choose_value_or_infer_conflict():
    results = [fragment("a", text="甲型设备三天", effective_date="2020-01-01", authority="企业"),
               fragment("b", text="乙型设备十五天", effective_date="2026-01-01", authority="强制性国家标准")]
    before = deepcopy(results)
    annotated = annotate_conflicts(results, {})
    assert [row["conflict_status"] for row in annotated] == ["none", "none"]
    assert [row["text"] for row in annotated] == [row["text"] for row in results]
    assert all(row["resolution"] == [] and row["conflict_ids"] == [] for row in annotated)
    assert results == before


def test_explicit_conflict_preserves_each_side_and_reference_locations():
    results = [fragment("a", page=8), fragment("b", page=21), fragment("unrelated")]
    annotated = annotate_conflicts(results, {"conflicts": [conflict()]})
    assert [row["conflict_status"] for row in annotated] == ["open", "open", "none"]
    assert annotated[0]["conflict_ids"] == ["conflict-1"]
    details = annotated[0]["conflicts"][0]
    assert details["reason"] == "已核对原文周期不一致"
    assert details["applicability_condition"] == "同一甲烷传感器型号及使用场景"
    assert details["references"] == [citation(results[0]), citation(results[1])]
    assert annotated[0]["resolution"] == []


def test_registered_conflict_is_visible_even_if_only_one_side_recalled():
    result = annotate_conflicts([fragment("a")], {"conflicts": [conflict()]})[0]
    assert result["conflict_status"] == "open"
    assert result["conflicts"][0]["references"][1] == {"reference_id": "b"}


def test_entry_version_locator_selector_is_exact_and_unrelated_version_is_clean():
    selectors = [{"entry_id": "doc", "version": "v1", "locator": {"page": 8}},
                 {"entry_id": "doc", "version": "v1", "locator": {"page": 21}}]
    registration = conflict(fragments=selectors)
    registration.pop("reference_ids")
    results = [fragment("a", page=8), fragment("b", page=21), fragment("c", version="v2", page=8),
               fragment("d", page=9)]
    result = annotate_conflicts(results, {"conflicts": [registration]})
    assert [row["conflict_status"] for row in result] == ["open", "open", "none", "none"]


def test_resolved_conflict_requires_registered_resolution_and_retains_originals():
    resolution = {"entry_id": "review-record", "version": "approved-v1", "locator": {"section": "2"}}
    results = [fragment("a", text="三天"), fragment("b", text="十五天")]
    annotated = annotate_conflicts(results, {"conflicts": [conflict(status="resolved", resolution_reference=resolution)]})
    assert all(row["conflict_status"] == "resolved" for row in annotated)
    assert annotated[0]["resolution"] == [{"conflict_id": "conflict-1", "resolution_reference": resolution}]
    assert [row["text"] for row in annotated] == ["三天", "十五天"]


def test_open_conflict_takes_precedence_over_resolved_registration():
    metadata = {"conflicts": [conflict(status="resolved", resolution_reference={"reference_id": "decision"}),
                              conflict(conflict_id="conflict-2")]}
    result = annotate_conflicts([fragment("a"), fragment("b")], metadata)[0]
    assert result["conflict_status"] == "open"
    assert result["conflict_ids"] == ["conflict-1", "conflict-2"]
    assert result["resolution"][0]["conflict_id"] == "conflict-1"


def test_only_confirmed_replacement_chain_is_exposed_without_hiding_old_citations():
    def replacement(rid, old, new, confirmed=True):
        return {"replacement_id": rid, "predecessor": {"reference_id": old},
                "successor": {"reference_id": new}, "confirmed": confirmed,
                "resolution_reference": {"reference_id": "official-notice"}}
    metadata = {"replacements": [replacement("replacement-1", "a", "b"),
                                  replacement("replacement-2", "b", "c"),
                                  replacement("not-confirmed", "c", "d", False)]}
    results = [fragment(ref, text=ref) for ref in ("a", "b", "c", "d")]
    annotated = annotate_conflicts(results, metadata)
    assert [row["conflict_status"] for row in annotated] == ["superseded", "superseded", "resolved", "none"]
    assert [row["text"] for row in annotated] == ["a", "b", "c", "d"]
    assert annotated[0]["resolution"][0]["replacement_id"] == "replacement-1"
    assert annotated[0]["resolution"][0]["references"] == [citation(results[0]), citation(results[1])]


def numeric(ref, value, *, key="sensor-a:alarm", condition="甲型设备/回风巷", unit="%CH4"):
    return {"reference_id": ref, "comparison_key": key, "value": value, "unit": unit,
            "applicability_condition": condition}


def test_same_metric_device_condition_and_unit_numeric_difference_is_only_possible():
    metadata = {"numeric_claims": [numeric("a", "0.8"), numeric("b", "1.2")]}
    results = [fragment("a", text="0.8%"), fragment("b", text="1.2%")]
    annotated = annotate_conflicts(results, metadata)
    assert [row["conflict_status"] for row in annotated] == ["possible_conflict"] * 2
    assert annotated[0]["conflicts"][0]["status"] == "possible_conflict"
    assert annotated[0]["conflict_ids"] == annotated[1]["conflict_ids"]
    assert annotated[0]["resolution"] == []
    assert [row["text"] for row in annotated] == ["0.8%", "1.2%"]
    assert annotate_conflicts(results[::-1], metadata)[1]["conflict_ids"] == annotated[0]["conflict_ids"]


@pytest.mark.parametrize("change", [{"key": "sensor-b:alarm"}, {"condition": "乙型设备/回风巷"}, {"unit": "ppm"}])
def test_different_device_scene_or_unit_does_not_infer_numeric_conflict(change):
    metadata = {"numeric_claims": [numeric("a", "0.8"), numeric("b", "1.2", **change)]}
    assert all(row["conflict_status"] == "none" for row in annotate_conflicts([fragment("a"), fragment("b")], metadata))


def test_equivalent_numbers_and_repeated_fragment_claims_do_not_create_conflict():
    metadata = {"numeric_claims": [numeric("a", "0.80"), numeric("b", 0.8), numeric("a", "1.2")]}
    result = annotate_conflicts([fragment("a"), fragment("b")], {"numeric_claims": metadata["numeric_claims"][:2]})
    assert all(row["conflict_status"] == "none" for row in result)
    assert annotate_conflicts([fragment("a")], {"numeric_claims": [numeric("a", 3), numeric("a", 15)]})[0]["conflict_status"] == "none"


def test_annotation_does_not_mutate_nested_registration_or_results_and_is_repeatable():
    results = [fragment("a"), fragment("b")]
    metadata = {"conflicts": [conflict()]}
    before = deepcopy((results, metadata))
    first = annotate_conflicts(results, metadata)
    assert annotate_conflicts(first, metadata) == first
    first[0]["conflicts"][0]["references"][0]["locator"]["page"] = 999
    assert (results, metadata) == before


@pytest.mark.parametrize("record", [conflict(status="resolved"), conflict(status="invalid"),
                                    conflict(reference_ids=["a"]), conflict(reason="")])
def test_invalid_explicit_registration_is_rejected_instead_of_silently_resolved(record):
    with pytest.raises(ValueError):
        annotate_conflicts([fragment("a"), fragment("b")], {"conflicts": [record]})


def test_empty_result_has_no_conflict_and_numeric_unmatched_claim_is_ignored():
    assert annotate_conflicts([], {}) == []
    assert annotate_conflicts([fragment("a")], {"numeric_claims": [numeric("a", 3), numeric("not-recalled", 15)]})[0]["conflict_status"] == "none"


def test_locator_subset_matches_recalled_extra_coordinates_and_keeps_registry_fields():
    registration = conflict(fragments=[{"entry_id": "doc", "version": "v1", "locator": {"page": 8, "slice": {"start": 2}}},
                                       {"entry_id": "doc", "version": "v1", "locator": {"page": 21}}],
                            disclosure="local_only", source_issue_id="source-issue-1",
                            source_sha256="synthetic-source-sha")
    registration.pop("reference_ids")
    first = fragment("a", page=8)
    first["locator"].update(slice={"start": 2, "end": 5}, paragraph=4)
    other = fragment("wrong-slice", page=8)
    other["locator"]["slice"] = {"start": 9}
    result = annotate_conflicts([first, fragment("b", page=21), other], {"conflicts": [registration]})
    assert [row["conflict_status"] for row in result] == ["open", "open", "none"]
    assert result[0]["conflicts"][0]["disclosure"] == "local_only"
    assert result[0]["conflicts"][0]["source_sha256"] == "synthetic-source-sha"
    assert result[0]["conflicts"][0]["source_issue_id"] == "source-issue-1"
    assert result[0]["conflicts"][0]["references"][0]["locator"] == first["locator"]


def test_confirmed_replacement_preserves_disclosure_and_audit_fields():
    record = {"replacement_id": "replace-1", "predecessor": {"reference_id": "a"},
              "successor": {"reference_id": "b"}, "confirmed": True,
              "resolution_reference": {"reference_id": "notice"}, "disclosure": "public_normative",
              "reviewed_by": "synthetic-reviewer", "reason": "明确替代条款"}
    result = annotate_conflicts([fragment("a"), fragment("b")], {"replacements": [record]})
    assert result[0]["resolution"][0]["disclosure"] == "public_normative"
    assert result[0]["resolution"][0]["reviewed_by"] == "synthetic-reviewer"
    assert result[0]["resolution"][0]["reason"] == "明确替代条款"


@pytest.mark.parametrize("record", [
    conflict(reference_ids="ab"),
    {key: value for key, value in conflict(fragments=[{"entry_id": "doc", "version": "v1", "locator": {}},
                                                    {"entry_id": "doc", "version": "v1", "locator": {"page": 21}}]).items()
     if key != "reference_ids"},
])
def test_registration_rejects_string_id_list_and_empty_locator_that_would_match_whole_document(record):
    with pytest.raises(ValueError):
        annotate_conflicts([fragment("a"), fragment("b")], {"conflicts": [record]})
