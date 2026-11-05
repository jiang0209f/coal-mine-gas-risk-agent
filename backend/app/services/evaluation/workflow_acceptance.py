"""Registration-specific workflow checks, separate from historical report metrics."""
from collections import Counter
from datetime import datetime
import re

__all__ = ["build_acceptance"]

_MISSING = object()


def _path(value, path):
    for key in path.split("."):
        if isinstance(value, dict):
            value = value.get(key, _MISSING)
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            return _MISSING
    return value


def _nonempty(value):
    return value is not _MISSING and value is not None and value != "" and value != [] and value != {}


def _metric(rows, key, threshold=None):
    numerator = sum(bool(row[key]) for row in rows)
    denominator = len(rows)
    rate = numerator / denominator if denominator else None
    result = {"numerator": numerator, "denominator": denominator, "rate": rate}
    if threshold is not None:
        result.update(required_rate=threshold, requirement_met=rate is not None and rate >= threshold)
    return result


def _tasks(tasks):
    identifiers = [row.get("task_id") for row in tasks if isinstance(row.get("task_id"), str) and row["task_id"]]
    counts = Counter(identifiers)
    results, seen = [], set()
    for index, task in enumerate(tasks):
        identifier = task.get("task_id")
        if identifier and identifier in seen:
            continue
        if identifier:
            seen.add(identifier)
        expected = task.get("expected_tools", [])
        actual = {row.get("name", row.get("tool")) for row in task.get("tool_calls", task.get("tool_steps", []))
                  if isinstance(row, dict) and row.get("status") == "success"}
        required = list(dict.fromkeys(value for value in expected if isinstance(value, str) and value)) if isinstance(expected, list) else []
        missing = [name for name in required if name not in actual]
        outputs = task.get("expected_outputs")
        response = task.get("actual_output", task.get("output", _MISSING))
        checks = []
        if isinstance(outputs, dict):
            for path, expected_value in outputs.items():
                actual_value = _path(response, path) if isinstance(path, str) else _MISSING
                checks.append({"path": path, "passed": actual_value is not _MISSING and actual_value == expected_value})
        elif isinstance(outputs, list):
            checks = [{"path": path, "passed": isinstance(path, str) and _nonempty(_path(response, path))} for path in outputs]
        complete = bool(identifier and required and not missing and checks and all(row["passed"] for row in checks)
                        and counts.get(identifier) == 1)
        results.append({"task_id": identifier or f"missing-task-id-{index}", "expected_tools": required,
                        "successful_tools": sorted(name for name in actual if isinstance(name, str)), "missing_tools": missing,
                        "output_checks": checks, "complete": complete, "duplicate_id": counts.get(identifier, 0) > 1})
    result = _metric(results, "complete")
    result.update(independent_task_count=len(counts), minimum_independent_tasks=21,
                  independence_check="unique_task_ids", duplicate_task_ids=sorted(key for key, count in counts.items() if count > 1),
                  task_count_requirement_met=len(counts) >= 21 and all(count == 1 for count in counts.values()),
                  all_expected_tasks_completed=bool(results) and all(row["complete"] for row in results), per_task=results)
    return result


def _time(value):
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        return moment if isinstance(moment, datetime) and moment.utcoffset() is not None else None
    except (ValueError, TypeError):
        return None


def _window(row):
    window = row.get("time_window") or {}
    start = _time(window.get("start_at", row.get("start_at"))) if isinstance(window, dict) else None
    end = _time(window.get("end_at", row.get("end_at") or row.get("revision_as_of") or row.get("as_of"))) if isinstance(window, dict) else None
    return bool(start and end and start <= end)


def _observation(reference):
    return (isinstance(reference, dict) and bool(reference.get("observation_id", reference.get("id")))
            and bool(reference.get("source_file_id")) and type(reference.get("row_number")) is int and reference["row_number"] >= 1)


def _model_or_rule(reference):
    return isinstance(reference, dict) and (bool(reference.get("run_id")) or bool(reference.get("rule_version")))


def _event_evidence(events, reports):
    latest = {}
    for index, event in enumerate(events):
        identifier = event.get("event_id") or f"missing-event-id-{index}"
        old = latest.get(identifier)
        if old is None or (event.get("event_revision") or 0) > (old.get("event_revision") or 0):
            latest[identifier] = event
    results = []
    for identifier, event in latest.items():
        related = [report for report in reports if report.get("event_id") == event.get("event_id")
                   and report.get("event_revision") == event.get("event_revision")
                   and all(not (event.get(key) and report.get(key)) or event[key] == report[key]
                           for key in ("source_id", "session_id", "face_id"))]
        evidence = [event, *related]
        observations, model_rules = [], []
        for row in evidence:
            observations.extend(row.get("observation_refs", []))
            model_rules.extend(row.get("model_or_rule_refs", []))
            common = row.get("evidence_refs", [])
            observations.extend(reference for reference in common if _observation(reference))
            model_rules.extend(reference for reference in common if _model_or_rule(reference))
        present = {"point": any(_nonempty(row.get("point_id")) for row in evidence),
                   "time_window": any(_window(row) for row in evidence),
                   "observation_refs": any(_observation(reference) for reference in observations),
                   "model_or_rule_refs": any(_model_or_rule(reference) for reference in model_rules)}
        missing = [key for key, exists in present.items() if not exists]
        results.append({"event_id": identifier, "event_revision": event.get("event_revision"),
                        "association_checks": present, "missing_fields": missing,
                        "complete": bool(event.get("event_id")) and not missing,
                        "report_ids": [row.get("report_id") for row in related if row.get("report_id")]})
    result = _metric(results, "complete", .90)
    result.update(denominator_unit="events", event_revision_policy="latest_supplied_revision_per_event", per_event=results)
    return result


def _locator(value):
    return isinstance(value, dict) and any(
        (type(item) is int and item >= 0) or (isinstance(item, str) and bool(item.strip()))
        for key, item in value.items() if key in {"page", "physical_page", "paragraph", "line", "row", "clause", "article", "section"})


def _normalize(value):
    return re.sub(r"\s+", "", value) if isinstance(value, str) else ""


def _measure_coverage(plans):
    rows = []
    for plan in plans:
        for index, item in enumerate(plan.get("items", [])):
            facts = item.get("review_facts") if isinstance(item.get("review_facts"), dict) else {}
            measure_state = item.get("measure_current_state", item.get("measure_state", facts.get("measure_state", item.get("state"))))
            reference_state = item.get("reference_current_state", item.get("reference_state", facts.get("reference_state")))
            applicability = item.get("applicability_check", facts.get("applicability_check"))
            if isinstance(applicability, dict):
                applicable = applicability.get("status") == "passed"
                applicability_method = applicability.get("method", "provided_scope_check")
            else:
                applicable = applicability == "passed"
                applicability_method = "provided_scope_check"
            text, source = _normalize(item.get("text")), _normalize(item.get("source_text", facts.get("source_text")))
            consistent, method = bool(text and source and text in source), "exact_excerpt"
            review = facts.get("content_consistency")
            if not consistent and isinstance(review, dict) and review.get("decision") == "consistent" and review.get("actor_id") and _time(review.get("reviewed_at")):
                consistent, method = True, "human_review"
            checks = {"measure_currently_approved": measure_state == "approved", "reference_currently_approved": reference_state == "approved",
                      "reference_id": _nonempty(item.get("reference_id")), "original_locator": _locator(item.get("locator")),
                      "applicable_conditions": applicable, "content_supported": consistent}
            rows.append({"plan_id": plan.get("plan_id"), "item_index": index, "measure_id": item.get("measure_id"),
                         "reference_id": item.get("reference_id"), "checks": checks,
                         "missing_fields": [key for key, value in checks.items() if not value],
                         "consistency_method": method, "applicability_method": applicability_method,
                         "complete": all(checks.values())})
    result = _metric(rows, "complete", .95)
    result.update(denominator_unit="plan_measure_items", per_measure=rows,
                  consistency_scope="exact_excerpt_or_explicit_human_review; no automatic semantic consistency claim")
    return result


def build_acceptance(tasks, events, reports, plans):
    """Compute counts only from supplied records and explicit expected results.

    Task expected_outputs is a path/value mapping, or paths required to be nonempty;
    actual_output is the actual saved service response. Plan items supply current
    review facts, original source_text, locator and an explicit applicability check.
    This function performs no retrieval, approvals, model calls or database writes.
    """
    for rows in (tasks, events, reports, plans):
        if not isinstance(rows, (list, tuple)) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("验收输入须为记录对象列表")
    return {"schema_version": "workflow-acceptance-v1", "tool_collaboration": _tasks(tasks),
            "evidence_completeness": _event_evidence(events, reports), "measure_basis_coverage": _measure_coverage(plans)}
