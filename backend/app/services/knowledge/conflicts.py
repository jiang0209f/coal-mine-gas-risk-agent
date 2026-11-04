"""对召回条文附加登记分歧、数值差异提示及已确认替代关系。"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json

__all__ = ["annotate_conflicts"]

_PRIORITY = {"none": 0, "resolved": 1, "superseded": 2, "possible_conflict": 3, "open": 4}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _selector(value):
    if not isinstance(value, dict):
        raise ValueError("引用定位必须为字典")
    if _text(value.get("reference_id")):
        return {"reference_id": value["reference_id"]}
    if (_text(value.get("entry_id")) and value.get("version") is not None
            and str(value["version"]).strip() and value.get("locator")
            and isinstance(value["locator"], (dict, str))):
        return {key: deepcopy(value[key]) for key in ("entry_id", "version", "locator")}
    raise ValueError("引用需要reference_id或entry_id、version和locator")


def _matches(row, selector):
    if "reference_id" in selector:
        return row.get("reference_id") == selector["reference_id"]
    return (row.get("entry_id") == selector["entry_id"]
            and str(row.get("version")) == str(selector["version"])
            and _contains(row.get("locator"), selector["locator"]))


def _contains(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def _identity(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _references(selectors, rows):
    references = []
    for selector in selectors:
        matches = [row for row in rows if _matches(row, selector)]
        references.extend([{key: deepcopy(row[key]) for key in
                            ("reference_id", "entry_id", "version", "locator") if key in row}
                           for row in matches] or [deepcopy(selector)])
    return list({_identity(ref): ref for ref in references}.values())


def _annotate(row, status, *, detail=None, resolution=None):
    if _PRIORITY[status] > _PRIORITY[row["conflict_status"]]:
        row["conflict_status"] = status
    if detail is not None:
        if detail["conflict_id"] not in row["conflict_ids"]:
            row["conflict_ids"].append(detail["conflict_id"])
            row["conflicts"].append(deepcopy(detail))
    if resolution is not None:
        row["resolution"].append(deepcopy(resolution))


def annotate_conflicts(results: list[dict], document_metadata: dict | None) -> list[dict]:
    """返回附有分歧标记的副本，不改写、删除原文或自动确定数值。

    document_metadata包含三个可选列表：
    conflicts: conflict_id、reference_ids或fragments、topic、
        applicability_condition、reason、status(open/resolved)；resolved还需
        resolution_reference引用定位。entry_id/version精确绑定，locator按字典子集匹配。
    replacements: replacement_id、confirmed、predecessor、successor、
        resolution_reference；只使用confirmed为True的替代关系。
    numeric_claims: 引用定位、comparison_key、applicability_condition、unit、value；
        只比较同key/条件/单位的已召回不同引用，差值标为possible_conflict。

    元数据由调用方在授权范围内提供。日期、法律层级和未登记的原文数字不参与推断。
    resolution只复述登记的审核/替代依据；不选择阈值、标校周期或其他业务参数。
    """
    rows = deepcopy(results)
    metadata = document_metadata or {}
    for row in rows:
        row.update(conflict_status="none", conflict_ids=[], conflicts=[], resolution=[])
    seen = set()
    for record in metadata.get("conflicts", []):
        if (not all(_text(record.get(key)) for key in
                    ("conflict_id", "topic", "applicability_condition", "reason"))
                or record.get("status", "open") not in ("open", "resolved")):
            raise ValueError("冲突登记缺少编号、主题、适用条件、原因或有效状态")
        if record["conflict_id"] in seen:
            raise ValueError("冲突编号重复")
        seen.add(record["conflict_id"])
        if bool(record.get("reference_ids")) == bool(record.get("fragments")):
            raise ValueError("冲突需要reference_ids或fragments中的一种定位方式")
        if not isinstance(record.get("reference_ids") or record.get("fragments"), list):
            raise ValueError("冲突引用定位需要列表")
        selectors = ([_selector({"reference_id": ref}) for ref in record["reference_ids"]]
                     if record.get("reference_ids") else [_selector(ref) for ref in record["fragments"]])
        if len({_identity(selector) for selector in selectors}) < 2:
            raise ValueError("冲突登记需要至少两个不同引用定位")
        status = record.get("status", "open")
        resolution = None
        if status == "resolved":
            _selector(record.get("resolution_reference"))
            resolution = {"conflict_id": record["conflict_id"],
                          "resolution_reference": deepcopy(record["resolution_reference"])}
        detail = deepcopy(record)
        detail.update(status=status, references=_references(selectors, rows))
        for row in rows:
            if any(_matches(row, selector) for selector in selectors):
                _annotate(row, status, detail=detail, resolution=resolution)

    for record in metadata.get("replacements", []):
        if record.get("confirmed") is not True:
            continue
        if not _text(record.get("replacement_id")):
            raise ValueError("替代关系缺少编号")
        predecessor, successor = _selector(record.get("predecessor")), _selector(record.get("successor"))
        if _identity(predecessor) == _identity(successor):
            raise ValueError("替代关系的前后引用不能相同")
        _selector(record.get("resolution_reference"))
        resolution = deepcopy(record)
        resolution["references"] = _references([predecessor, successor], rows)
        for row in rows:
            if _matches(row, predecessor):
                _annotate(row, "superseded", resolution=resolution)
            elif _matches(row, successor):
                _annotate(row, "resolved", resolution=resolution)

    groups = {}
    for claim in metadata.get("numeric_claims", []):
        if not all(_text(claim.get(key)) for key in ("comparison_key", "applicability_condition", "unit")):
            continue
        try:
            value = Decimal(str(claim["value"]))
            selector = _selector(claim)
        except (KeyError, ValueError, InvalidOperation):
            continue
        if not value.is_finite() or not any(_matches(row, selector) for row in rows):
            continue
        key = tuple(claim[field] for field in ("comparison_key", "applicability_condition", "unit"))
        groups.setdefault(key, []).append((selector, value))
    for (topic, condition, unit), claims in groups.items():
        if len({value for _, value in claims}) < 2 or len({_identity(selector) for selector, _ in claims}) < 2:
            continue
        selectors = list({_identity(selector): selector for selector, _ in claims}.values())
        fingerprint = _identity({"topic": topic, "condition": condition, "unit": unit,
                                "claims": sorted((_identity(selector), str(value.normalize())) for selector, value in claims)})
        detail = {"conflict_id": "possible:" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16],
                  "topic": topic, "applicability_condition": condition, "status": "possible_conflict",
                  "reason": "同一比较项、适用条件和单位的登记数值存在差异，需核对原文及适用范围",
                  "references": _references(selectors, rows)}
        for row in rows:
            if any(_matches(row, selector) for selector in selectors):
                _annotate(row, "possible_conflict", detail=detail)
    return rows
