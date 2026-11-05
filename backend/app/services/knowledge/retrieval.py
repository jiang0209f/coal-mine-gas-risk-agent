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
    "断电": ("断电", "切断电源", "断开电源", "停电", "停止全部电路", "停止所有电路", "停止工作面所有电路", "停止工作面全部电路"),
    "恢复供电": ("恢复供电", "恢复送电", "重新送电", "重新供电", "复电", "送电", "合闸"),
    "撤离": ("撤离", "撤人", "撤出人员", "人员撤出", "疏散", "避灾路线", "逃生路线"),
    "调校": ("调校", "标校", "校准", "标定", "标气", "用气样", "气样校表", "calibration", "calibrate"),
    "抽采": ("抽采", "抽放"),
    "通风": ("通风", "换气", "风路", "主扇", "主风机", "大风机", "主要通风机", "ventilation"),
    "复工": ("复工", "复产", "恢复作业", "恢复生产"),
    "应急预案": ("应急预案", "现场处置方案", "应急处置"),
    "预测": ("预测", "预报", "forecast"),
    "预警": ("预警", "告警", "报警", "warning"),
    "溯源": ("溯源", "成因", "causation"),
    "主机热备": ("双机热备", "备用主机", "热备"),
    "断线补传": ("补传", "重连补数", "历史补数"),
    "备份恢复": ("备份恢复", "数据恢复", "恢复验证", "验证能否恢复"),
    "记录更正": ("覆盖旧数", "覆盖原值", "修改原值", "改写原始数据", "修改原始数据"),
}
_GENERIC_CONCEPTS = {"瓦斯", "传感器"}
_MIN_COSINE = 0.60
_MIN_JOINT_COSINE = 0.55
_RERANK_POOL_SIZE = 50
_MIN_RERANK_SCORE = 0.05
_RRF_OFFSET = 60


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
    concepts = {concept for concept, aliases in _CONCEPTS.items() if any(contains(alias) for alias in aliases)}
    if (any(word in text for word in ("遇险", "事故", "灾害", "爆炸", "火灾"))
            and any(word in text for word in ("遇险", "发生", "突发", "遭遇", "井下"))
            and any(word in text for word in ("离开", "出去", "逃生", "逃离", "脱险"))):
        concepts.add("撤离")
    if ("电源" in text and "接通" in text
            and any(word in text for word in ("瓦斯", "甲烷", "浓度", "读数", "断电", "停电", "复电"))):
        concepts.add("恢复供电")
    if "还原" in text and any(word in text for word in ("数据库", "备份", "拷贝", "副本", "本机库", "监测")):
        concepts.add("备份恢复")
    if (re.search(r"备用机(?!组)", text)
            and any(word in text for word in ("监控", "电脑", "主机", "中心站", "计算机", "双机"))):
        concepts.add("主机热备")
    if ("补来的" in text
            and any(word in text for word in ("当前值", "测值", "数值", "读数", "数据", "采样", "补数"))):
        concepts.add("断线补传")
    if (any(word in text for word in ("更正", "新修订"))
            and any(word in text for word in ("数据", "监测", "原始记录", "原记录", "采样", "点位", "读数", "旧数", "原值", "曲线"))):
        concepts.add("记录更正")
    return concepts


def _supply_types(text):
    aliases = {"power": ("供电", "送电", "复电", "电源", "电能", "合闸"),
               "gas": ("供气",), "compressed_air": ("压风", "压缩空气")}
    return {kind for kind, words in aliases.items() if any(word in text for word in words)}


def _fan_types(text):
    aliases = {"main": ("主扇", "主风机", "大风机", "主要通风机"),
               "local": ("局扇", "局部通风机", "掘进风机")}
    return {kind for kind, words in aliases.items() if any(word in text for word in words)}


def _inferred_topics(text):
    """Infer search topics from object/action combinations, keeping original text."""
    def has(words):
        return any(word in text for word in words)
    topics = set()
    if "新鲜风" in text or ("迎头" in text and has(("送", "输风", "空气输送"))):
        topics.add("通风")
    measurement = has(("示值", "读数", "测值", "气样", "标准气", "误差", "浓度"))
    placement = has(("位置", "吊挂", "高度", "搬移", "移位", "安设", "固定安装", "角度"))
    delivery = has(("型号", "包装", "数量", "附件", "库存", "到货"))
    if (has(("传感器", "探头", "测头")) and not placement
            and ((measurement and has(("调", "校", "标", "比对", "对照")))
                 or (not delivery and has(("调好", "调过", "调校", "校准", "标校", "标定", "比对信息"))))):
        topics.add("调校")
    if "还原" in text and "完整性" in text and has(("时点", "时间点", "恢复点", "版本")):
        topics.add("备份恢复")
    audit_axes = (has(("谁", "何人", "操作者")), has(("何时", "什么时候", "时间", "时刻")),
                  has(("因何", "原因", "理由", "为何")))
    if (has(("测值", "测量值", "读数", "数值", "监测值", "采样值", "数字", "某个数", "这个数"))
            and has(("修改", "改动", "改写", "更正", "变更", "改的", "改了")) and sum(audit_axes) >= 2):
        topics.add("记录更正")
    if (has(("巷道", "钻孔", "煤层")) and has(("气体", "瓦斯", "甲烷"))
            and has(("孔内", "孔里", "钻孔")) and has(("抽取", "抽出", "抽排", "抽得", "预抽", "抽采", "抽放"))
            and has(("达标", "合格", "达效", "评判"))):
        topics.add("抽采")
    if has(("能源", "供能")) and "恢复" in text and not _supply_types(text):
        topics.add("能源恢复")
    return topics


def _has_business_context(query):
    text = _normalize(query)
    if _concepts(text) or _inferred_topics(text):
        return True
    return any(word in text for word in (
        "煤矿", "矿井", "工作面", "巷道", "煤层", "井下", "钻孔", "浓度", "风量", "风速",
        "监测", "测点", "点位", "联锁", "闭锁", "隔爆", "气样", "采样", "采集",
        "数据", "记录", "日志", "原值", "旧数", "读数", "测值", "数值", "误差",
        "上送", "重发", "回执", "备份", "拷贝", "快照", "版本", "时点", "签批", "审批",
        "复核", "处置", "规程", "条文", "法规", "标准", "模型", "回放", "评估", "阈值"))


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


def _ranking_key(item):
    return (-item[0]["score"], item[0]["reference_id"], item[1], str(item[0]["version"]))


def _rerank(query, original, semantic_candidates, reranker, limit):
    original_ids = {item[0]["reference_id"] for item in original}
    business_context = _has_business_context(query)
    dense = sorted(semantic_candidates,
                   key=lambda item: (-item[0]["semantic_score"], *_ranking_key(item)))
    dense_ranks = {item[0]["reference_id"]: rank for rank, item in enumerate(dense, 1)}
    pool = {item[0]["reference_id"]: item for item in [*original[:_RERANK_POOL_SIZE], *dense[:_RERANK_POOL_SIZE]]}
    if not pool:
        return []
    entries = list(pool.values())
    status = "available"
    try:
        scores = reranker(query, [item[0]["title"] + "\n" + item[0]["text"] for item in entries])
    except Exception as error:
        status = "error:" + type(error).__name__
    else:
        try:
            scores = list(scores)
            if (len(scores) != len(entries)
                    or any(isinstance(value, bool) or not isinstance(value, Real)
                           or not math.isfinite(value) or not 0 <= value <= 1 for value in scores)):
                raise ValueError("reranker scores")
        except Exception:
            status = "invalid_scores"
    if status != "available":
        warnings.warn("检索重排不可用，已退回原混合检索", RuntimeWarning, stacklevel=3)
        return [{**item[0], "reranker_status": status, "reranking_mode": "hybrid_fallback"}
                for item in original[:limit]]
    rerank_order = sorted(range(len(entries)), key=lambda index: (-scores[index], *_ranking_key(entries[index])))
    rerank_ranks = {entries[index][0]["reference_id"]: rank for rank, index in enumerate(rerank_order, 1)}
    original_ranks = {item[0]["reference_id"]: rank for rank, item in enumerate(original, 1)}
    found = []
    for (candidate, locator), relevance in zip(entries, scores):
        existing = candidate["reference_id"] in original_ids
        if not existing and (relevance < _MIN_RERANK_SCORE or not business_context):
            continue
        reference = candidate["reference_id"]
        semantic_rank = dense_ranks.get(reference)
        retrieval_rank = original_ranks.get(reference, semantic_rank)
        rank_source = "hybrid" if existing and semantic_rank is not None else "semantic" if semantic_rank is not None else "lexical"
        rank_label = {"hybrid": "混合", "semantic": "语义", "lexical": "词法"}[rank_source]
        rerank_rank = rerank_ranks[reference]
        fusion = (1 / (_RRF_OFFSET + retrieval_rank) + 1 / (_RRF_OFFSET + rerank_rank)) / (2 / (_RRF_OFFSET + 1))
        score = 0.9 + 0.1 * fusion if candidate["exact_matches"] else 0.85 * fusion
        found.append(({**candidate, "score": score, "hybrid_score": candidate["score"],
                       "rerank_score": relevance, "reranker_status": status,
                       "reranking_mode": "cross_encoder",
                       "semantic_rank": semantic_rank, "rerank_rank": rerank_rank,
                       "retrieval_rank": retrieval_rank, "retrieval_rank_source": rank_source,
                       "fusion_method": "reciprocal_rank",
                       "recall_path": "original_hybrid" if existing else "semantic_pool",
                       "rerank_reasons": [*candidate["rerank_reasons"], f"相关性重排：{relevance:.3f}",
                                          f"排名融合：{rank_label}第{retrieval_rank}，重排第{rerank_rank}"]}, locator))
    found.sort(key=lambda item: (-item[0]["score"], -item[0]["rerank_score"],
                                item[0]["reference_id"], item[1], str(item[0]["version"])))
    return [item[0] for item in found[:limit]]


def retrieve(query: str, candidates: Sequence[Mapping], embedding: Callable | None = None,
             limit: int = 50, *, reranker: Callable | None = None) -> list[dict]:
    """Return relevant copied candidates, scores and inspectable ranking reasons.

    Exact standard editions and requested articles/sections constrain retrieval.
    Otherwise lexical coverage, bounded domain synonyms and cosine >=0.60 recall
    passages. Weak lexical overlaps without an identifier, domain concept or
    continuous phrase need cosine >=0.55 when vectors are available. These are
    initial retrieval thresholds; scores in [0,1] are ranking signals.
    Invalid embeddings emit a warning and use a documented lexical fallback.
    When configured, a local reranker receives the union of the top 50 original
    and top 50 dense candidates before the old cosine gate. New dense-only
    candidates require reranker support; failed reranking returns only the
    original results. Access and exact identifier constraints apply to both.
    """
    if not isinstance(query, str):
        raise ValueError("检索词必须为字符串")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("limit必须为1至50的整数")
    if embedding is not None and not callable(embedding):
        raise ValueError("embedding必须为可调用的本地向量接口")
    if reranker is not None and not callable(reranker):
        raise ValueError("reranker必须为可调用的本机相关性接口")
    prepared = _candidates(candidates)
    if not query.strip() or not prepared:
        return []
    normalized_query = _normalize(query.strip())
    terms, concepts = _terms(normalized_query), _concepts(normalized_query)
    inferred_topics = _inferred_topics(normalized_query) - concepts
    concepts.update(inferred_topics)
    energy_unspecified = "能源恢复" in inferred_topics
    query_fans, query_supply = _fan_types(normalized_query), _supply_types(normalized_query)
    long_phrases = _long_phrases(normalized_query)
    exact_body_phrase = (bool(re.fullmatch(r"[\u4e00-\u9fff]{4,}", normalized_query))
                         and any(normalized_query in _normalize(row[0]["text"]) for row in prepared))
    wants_experience = "经验" in normalized_query and not any(word in normalized_query for word in ("条文", "规程", "法规", "标准"))
    requested = _identifiers(normalized_query, query=True)
    semantic_scores, embedding_status = _semantic_scores(query, prepared, embedding)
    mode = "hybrid" if embedding_status == "available" else ("lexical" if embedding is None else "lexical_fallback")
    found, semantic_candidates = [], []
    for (candidate, text, locator_key), semantic in zip(prepared, semantic_scores):
        if wants_experience and candidate.get("kind") != "experience" and "经验" not in _normalize(candidate["title"] + candidate["text"]):
            continue
        identifiers = _identifiers(text)
        exact_groups = [wanted & available for wanted, available in zip(requested, identifiers)]
        if any(wanted and not matched for wanted, matched in zip(requested, exact_groups)):
            continue
        exact = sorted(set().union(*exact_groups))
        if not exact:
            candidate_fans, candidate_supply = _fan_types(text), _supply_types(text)
            if query_fans and candidate_fans and query_fans.isdisjoint(candidate_fans):
                continue
            if query_supply and candidate_supply and query_supply.isdisjoint(candidate_supply):
                continue
        if (exact_body_phrase and not exact and normalized_query not in _normalize(candidate["text"])
                and not (semantic is not None and semantic >= _MIN_COSINE)):
            continue
        matched_terms = sorted((term for term in terms if term in text), key=lambda term: (query.find(term), term))
        candidate_concepts = _concepts(text)
        if energy_unspecified and any(phrase in text for phrase in (
                "恢复供电", "恢复送电", "重新送电", "重新供电", "复电", "送电", "合闸",
                "恢复供气", "供气恢复", "重新供气", "恢复压风", "压风恢复")):
            candidate_concepts.add("能源恢复")
        matched_concepts = sorted(concepts & candidate_concepts)
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
        accepted = exact or lexical_signal or (semantic is not None and semantic >= _MIN_COSINE)
        if not accepted and reranker is None:
            continue
        base_score = lexical if semantic is None else 0.5 * lexical + 0.5 * semantic
        score = 0.9 + 0.1 * base_score if exact else 0.85 * base_score
        reasons = (["编号匹配：" + value for value in exact]
                   + ["同义词匹配：" + value for value in matched_concepts])
        if matched_phrases:
            reasons.append("连续词组匹配：" + matched_phrases[0])
        if matched_terms:
            reasons.append(f"关键词覆盖率：{coverage:.3f}")
        if semantic is not None:
            reasons.append(f"向量相似度：{semantic:.3f}")
        if inferred_topics:
            reasons.append("组合主题识别：" + "、".join(sorted(inferred_topics)))
        if energy_unspecified:
            reasons.append("供能对象待明确：并列检索供电、供气和压风要求")
        item = ({**candidate, "score": score, "lexical_score": lexical,
                 "semantic_score": semantic, "matched_terms": matched_terms,
                 "matched_concepts": matched_concepts, "exact_matches": exact,
                 "rerank_reasons": reasons, "retrieval_mode": mode,
                 "embedding_status": embedding_status, "inferred_query_topics": sorted(inferred_topics),
                 "topic_ambiguities": ["energy_supply_type_unspecified"] if energy_unspecified else []}, locator_key)
        if accepted:
            found.append(item)
        if reranker is not None and semantic is not None:
            semantic_candidates.append(item)
    found.sort(key=_ranking_key)
    if reranker is not None:
        return _rerank(query, found, semantic_candidates, reranker, limit)
    return [item[0] for item in found[:limit]]
