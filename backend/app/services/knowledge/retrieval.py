"""Local, deterministic passage retrieval with optional injected embeddings.

The caller supplies candidates after applying access, approval and version rules.
An embedding callable accepts [query, *passage_texts] and returns matching vectors.
This module performs no model loading, network requests or database operations.
"""

from collections.abc import Callable, Mapping, Sequence
import json
import math
from numbers import Real
import re
import unicodedata
import warnings

__all__ = ["retrieve"]

_STANDARD = re.compile(r"(?<![a-z0-9])((?:gb|aq|ka|mt|nb|db\d{1,3})(?:\s*/\s*t)?\s*\d{3,6}(?:\.\d+)*\s*-\s*\d{4})(?!\d)", re.I)
_ARTICLE = re.compile(r"第\s*([\d〇零一二两三四五六七八九十百千]+)\s*条")
_SECTION = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){2,4})(?![\d.])")
_SHORT_SECTION = re.compile(r"(?:第\s*|条款\s*|条文\s*|clause\s*|section\s*)(\d{1,3}\.\d{1,3})(?:\s*[条款项]|(?![\d.]))", re.I)
_STOP = ("什么情况下", "啥时候", "什么时候", "怎么样", "怎么", "如何", "为什么", "为何", "请问", "请查", "是否", "哪些", "怎样", "多少", "可以", "应该", "需要", "进行", "要求", "规定", "关于", "相关", "煤矿", "矿井", "工作面", "的", "了", "吗", "呢", "中", "后")
_CONCEPTS = {
    "瓦斯": ("瓦斯", "甲烷", "methane", "gas"),
    "传感器": ("传感器", "探头", "测头", "sensor", "detector"),
    "超限": ("超限", "超标", "超过限值", "超过阈值", "浓度过高"),
    "断电": ("断电", "切断电源", "断开电源", "停电"),
    "恢复供电": ("恢复供电", "恢复送电", "重新送电", "重新供电", "复电", "送电"),
    "撤离": ("撤离", "撤人", "撤出人员", "人员撤出", "疏散"),
    "调校": ("调校", "标校", "校准", "标定", "calibration", "calibrate"),
    "抽采": ("抽采", "抽放"),
    "通风": ("通风", "换气", "ventilation"),
    "复工": ("复工", "复产", "恢复作业", "恢复生产"),
    "应急预案": ("应急预案", "现场处置方案", "应急处置"),
    "预测": ("预测", "预报", "forecast"),
    "预警": ("预警", "告警", "报警", "warning"),
    "溯源": ("溯源", "成因", "causation"),
}
_GENERIC_CONCEPTS = {"瓦斯", "传感器"}
_MIN_COSINE = 0.60
_MIN_JOINT_COSINE = 0.55


def _normalize(text):
    return unicodedata.normalize("NFKC", text).casefold().translate(str.maketrans({"—": "-", "–": "-", "−": "-"}))


def _chinese_number(value):
    if value.isdecimal():
        return int(value)
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    units = {"十": 10, "百": 100, "千": 1000}
    total, digit = 0, 0
    for character in value:
        if character in units:
            total += (digit or 1) * units[character]
            digit = 0
        else:
            digit = digits[character]
    return total + digit


def _identifiers(text, *, query=False):
    standards = {re.sub(r"\s+", "", match).upper() for match in _STANDARD.findall(text)}
    articles = {f"第{_chinese_number(match)}条" for match in _ARTICLE.findall(text)}
    sections = set(_SECTION.findall(text)) | set(_SHORT_SECTION.findall(text))
    if not query:
        sections.update(re.findall(r"(?<![\d.])(\d{1,3}\.\d{1,3})(?![\d.])", text))
    return standards, articles, sections


def _terms(query):
    text = _STANDARD.sub(" ", query)
    text = _ARTICLE.sub(" ", text)
    text = _SECTION.sub(" ", text)
    text = _SHORT_SECTION.sub(" ", text)
    for stop in _STOP:
        text = text.replace(stop, " ")
    terms = set(re.findall(r"[a-z][a-z0-9_]{1,}", text))
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(segment[index:index + 2] for index in range(len(segment) - 1))
    return terms


def _concepts(text):
    def contains(alias):
        if alias.isascii():
            return re.search(r"(?<![a-z0-9_])" + re.escape(alias) + r"(?![a-z0-9_])", text) is not None
        return alias in text
    return {concept for concept, aliases in _CONCEPTS.items() if any(contains(alias) for alias in aliases)}


def _long_phrases(query):
    for stop in _STOP:
        query = query.replace(stop, " ")
    return {segment[index:index + 4] for segment in re.findall(r"[\u4e00-\u9fff]+", query)
            for index in range(len(segment) - 3)}


def _candidates(candidates):
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("候选资料必须为列表或元组")
    prepared, seen = [], set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("候选资料必须包含引用编号、标题、正文、定位和版本")
        if any(not isinstance(candidate.get(key), str) or not candidate[key].strip()
               for key in ("reference_id", "title", "text")):
            raise ValueError("候选资料的引用编号、标题和正文必须为非空文本")
        if candidate["reference_id"] in seen:
            raise ValueError("候选资料的引用编号不能重复")
        seen.add(candidate["reference_id"])
        locator, version = candidate.get("locator"), candidate.get("version")
        if (not isinstance(locator, (str, dict)) or not locator
                or isinstance(version, bool) or not isinstance(version, (str, int)) or version == ""):
            raise ValueError("候选资料的定位或版本无效")
        try:
            locator_key = json.dumps(locator, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("候选资料的定位必须可序列化") from None
        source = candidate.get("source", "")
        if not isinstance(source, str) or re.search(r"(?:https?://|[a-zA-Z]:[\\/])", source):
            source = ""
        haystack = _normalize(" ".join((candidate["title"], candidate["text"], source, locator_key)))
        prepared.append((candidate, haystack, locator_key))
    return prepared


def _semantic_scores(query, prepared, embedding):
    if embedding is None:
        return [None] * len(prepared), "not_configured"
    try:
        batch = list(embedding([query, *(item[0]["title"] + "\n" + item[0]["text"] for item in prepared)]))
    except Exception as error:
        warnings.warn("检索向量不可用，已降级为词法检索", RuntimeWarning, stacklevel=3)
        return [None] * len(prepared), "error:" + type(error).__name__
    try:
        if len(batch) != len(prepared) + 1:
            raise ValueError("vector count")
        normalized = []
        dimensions = None
        for vector in batch:
            vector = list(vector)
            if not vector or (dimensions is not None and len(vector) != dimensions):
                raise ValueError("vector dimensions")
            if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in vector):
                raise ValueError("vector values")
            norm = math.hypot(*vector)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("vector norm")
            normalized.append([value / norm for value in vector])
            dimensions = len(vector)
        scores = [max(0.0, min(1.0, math.fsum(a * b for a, b in zip(normalized[0], vector))))
                  for vector in normalized[1:]]
        return scores, "available"
    except (TypeError, ValueError, OverflowError):
        warnings.warn("检索向量格式无效，已降级为词法检索", RuntimeWarning, stacklevel=3)
        return [None] * len(prepared), "invalid_vectors"


def retrieve(query: str, candidates: Sequence[Mapping], embedding: Callable | None = None,
             limit: int = 50) -> list[dict]:
    """Return relevant copied candidates, scores and inspectable ranking reasons.

    Exact standard editions and requested articles/sections constrain retrieval.
    Otherwise lexical coverage, bounded domain synonyms and cosine >=0.60 recall
    passages. Weak lexical overlaps without an identifier, domain concept or
    continuous phrase need cosine >=0.55 when vectors are available. These are
    initial retrieval thresholds; scores in [0,1] are ranking signals.
    Invalid embeddings emit a warning and use a documented lexical fallback.
    """
    if not isinstance(query, str):
        raise ValueError("检索词必须为字符串")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("limit必须为1至50的整数")
    if embedding is not None and not callable(embedding):
        raise ValueError("embedding必须为可调用的本地向量接口")
    prepared = _candidates(candidates)
    if not query.strip() or not prepared:
        return []
    normalized_query = _normalize(query.strip())
    terms, concepts = _terms(normalized_query), _concepts(normalized_query)
    long_phrases = _long_phrases(normalized_query)
    exact_body_phrase = (bool(re.fullmatch(r"[\u4e00-\u9fff]{4,}", normalized_query))
                         and any(normalized_query in _normalize(row[0]["text"]) for row in prepared))
    wants_experience = "经验" in normalized_query and not any(word in normalized_query for word in ("条文", "规程", "法规", "标准"))
    requested = _identifiers(normalized_query, query=True)
    semantic_scores, embedding_status = _semantic_scores(query, prepared, embedding)
    mode = "hybrid" if embedding_status == "available" else ("lexical" if embedding is None else "lexical_fallback")
    found = []
    for (candidate, text, locator_key), semantic in zip(prepared, semantic_scores):
        if wants_experience and candidate.get("kind") != "experience" and "经验" not in _normalize(candidate["title"] + candidate["text"]):
            continue
        identifiers = _identifiers(text)
        exact_groups = [wanted & available for wanted, available in zip(requested, identifiers)]
        if any(wanted and not matched for wanted, matched in zip(requested, exact_groups)):
            continue
        exact = sorted(set().union(*exact_groups))
        if (exact_body_phrase and not exact and normalized_query not in _normalize(candidate["text"])
                and not (semantic is not None and semantic >= _MIN_COSINE)):
            continue
        matched_terms = sorted((term for term in terms if term in text), key=lambda term: (query.find(term), term))
        matched_concepts = sorted(concepts & _concepts(text))
        matched_phrases = sorted(phrase for phrase in long_phrases if phrase in text)
        coverage = len(matched_terms) / len(terms) if terms else 0.0
        concept_coverage = len(matched_concepts) / len(concepts) if concepts else 0.0
        lexical = 0.55 * coverage + 0.45 * concept_coverage if concepts else coverage
        if exact:
            lexical = max(lexical, 0.9)
        specific = set(matched_concepts) - _GENERIC_CONCEPTS
        lexical_signal = (coverage >= 0.5 or bool(matched_phrases) or bool(specific)
                          or (bool(matched_concepts) and len(terms) <= 3))
        needs_joint_support = not exact and not matched_concepts and not matched_phrases and coverage < 0.75
        if needs_joint_support and semantic is not None and semantic < _MIN_JOINT_COSINE:
            lexical_signal = False
        if not exact and not lexical_signal and not (semantic is not None and semantic >= _MIN_COSINE):
            continue
        base_score = lexical if semantic is None else 0.6 * lexical + 0.4 * semantic
        score = 0.9 + 0.1 * base_score if exact else 0.85 * base_score
        reasons = (["编号匹配：" + value for value in exact]
                   + ["同义词匹配：" + value for value in matched_concepts])
        if matched_phrases:
            reasons.append("连续词组匹配：" + matched_phrases[0])
        if matched_terms:
            reasons.append(f"关键词覆盖率：{coverage:.3f}")
        if semantic is not None:
            reasons.append(f"向量相似度：{semantic:.3f}")
        found.append(({**candidate, "score": score, "lexical_score": lexical,
                       "semantic_score": semantic, "matched_terms": matched_terms,
                       "matched_concepts": matched_concepts, "exact_matches": exact,
                       "rerank_reasons": reasons, "retrieval_mode": mode,
                       "embedding_status": embedding_status}, locator_key))
    found.sort(key=lambda item: (-item[0]["score"], item[0]["reference_id"], item[1], str(item[0]["version"])))
    return [item[0] for item in found[:limit]]
