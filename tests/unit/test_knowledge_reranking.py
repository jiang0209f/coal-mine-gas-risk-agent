"""Two-stage retrieval recovers paraphrases without leaking a widened fallback."""
from importlib import import_module
import math

import pytest


def candidate(reference_id, text, title="合成条文"):
    return {"reference_id": reference_id, "title": title, "text": text,
            "locator": {"page": 1}, "version": "v1"}


@pytest.fixture
def retrieve():
    return import_module("app.services.knowledge.retrieval").retrieve


def vectors(*scores):
    return [[1, 0], *[[score, math.sqrt(1 - score ** 2)] for score in scores]]


def test_relevant_paraphrase_enters_pool_before_the_old_cosine_gate(retrieve):
    rows = [candidate("other", "设备维护安排与仪器检查。"),
            candidate("support", "对照基准测值，保存调整前后的结果与人员记录。")]
    calls = []
    def reranker(query, passages):
        calls.append((query, passages))
        return [0.1 if "设备维护" in text else 0.92 for text in passages]
    result = retrieve("读数纠偏后要留哪些信息", rows,
                      embedding=lambda texts: vectors(0.72, 0.49), reranker=reranker, limit=5)
    assert result[0]["reference_id"] == "support"
    assert result[0]["semantic_score"] == pytest.approx(0.49)
    assert result[0]["rerank_score"] == pytest.approx(0.92)
    assert result[0]["recall_path"] == "semantic_pool"
    assert result[0]["reranker_status"] == "available"
    assert len(calls) == 1 and calls[0][0] == "读数纠偏后要留哪些信息"
    assert len(calls[0][1]) == 2


def test_low_reranker_relevance_does_not_force_a_new_semantic_candidate(retrieve):
    result = retrieve("明天会下雨吗", [candidate("gas", "瓦斯超限应核查原因。")],
                      embedding=lambda texts: vectors(0.58),
                      reranker=lambda query, passages: [0.01])
    assert result == []


def test_supported_passage_can_have_a_modest_sigmoid_relevance(retrieve):
    rows = [candidate("restore", "在隔离环境中恢复备份，核对记录数量并记录验证结果。")]
    result = retrieve("如何确认数据备份能正常恢复", rows,
                      embedding=lambda texts: vectors(0.49),
                      reranker=lambda query, passages: [0.257])
    assert [row["reference_id"] for row in result] == ["restore"]
    assert result[0]["rerank_score"] == pytest.approx(0.257)


def test_dense_and_reranker_ranks_balance_a_semantically_distant_outlier(retrieve):
    rows = [candidate("support", "对照调整前后结果并保存记录。")]
    rows += [candidate(f"filler-{index}", f"合成正文编号{index}。") for index in range(17)]
    rows += [candidate("outlier-b", "其他业务附件。"), candidate("outlier-a", "其他业务台账。")]
    similarities = [0.95, *[0.90 - index * 0.02 for index in range(17)], 0.20, 0.10]
    def reranker(query, passages):
        return [0.99 if "业务台账" in text else 0.95 if "业务附件" in text
                else 0.89 if "调整前后" in text else 0.02 for text in passages]
    result = retrieve("仪器误差该怎样核验", rows, embedding=lambda texts: vectors(*similarities),
                      reranker=reranker, limit=5)
    assert result[0]["reference_id"] == "support"
    assert result[0]["semantic_rank"] == 1 and result[0]["rerank_rank"] == 3
    assert result[0]["fusion_method"] == "reciprocal_rank"


def test_strong_original_hybrid_match_is_not_replaced_by_a_weaker_dense_rank(retrieve):
    rows = [candidate("support", "甲烷传感器调校应保存记录。")]
    rows += [candidate(f"other-{index}", f"设备维护安排编号{index}。") for index in range(18)]
    similarities = [.2, *[.95 - index * .01 for index in range(18)]]
    def reranker(query, passages):
        return [.99 if "调校" in text else .8 for text in passages]
    result = retrieve("传感器调校", rows, embedding=lambda texts: vectors(*similarities),
                      reranker=reranker, limit=5)
    assert result[0]["reference_id"] == "support"
    assert result[0]["semantic_rank"] == 19 and result[0]["rerank_rank"] == 1
    assert result[0]["retrieval_rank_source"] == "hybrid" and result[0]["retrieval_rank"] == 1


@pytest.mark.parametrize("question", [
    "Python的Django项目跨域CORS报错怎样修", "如何还原手机出厂设置",
    "数学试卷的答案怎么更正", "电源适配器接通后为什么发热", "仓库补来的配件如何验收",
])
def test_widened_semantic_pool_requires_a_supported_business_context(retrieve, question):
    result = retrieve(question, [candidate("norm", "发生异常应核对标准程序，并保留核验信息。")],
                      embedding=lambda texts: vectors(.5042),
                      reranker=lambda query, passages: [.1311] * len(passages))
    assert result == []


def test_domain_context_also_supports_unmapped_data_action_paraphrases(retrieve):
    result = retrieve("上送的数据未回执能否重发", [candidate("support", "通信失败后应再次提交，并等待接收方反馈确认。")],
                      embedding=lambda texts: vectors(.49),
                      reranker=lambda query, passages: [.9] * len(passages))
    assert [row["reference_id"] for row in result] == ["support"]


def test_requested_standard_is_filtered_before_reranking(retrieve):
    rows = [candidate("right", "记录比对数据。", title="AQ 1029-2019"),
            candidate("old", "设备检查。", title="AQ 1029-2007"),
            candidate("other", "设备检查。", title="AQ 6201-2019")]
    calls = []
    def reranker(query, passages):
        calls.append(passages)
        return [0.01] * len(passages)
    result = retrieve("AQ1029-2019", rows, embedding=lambda texts: vectors(0.3, 0.95, 0.99),
                      reranker=reranker)
    assert [row["reference_id"] for row in result] == ["right"]
    assert len(calls) == 1 and len(calls[0]) == 1
    assert "2007" not in str(calls) and "6201" not in str(calls)


def test_reranker_failure_returns_only_the_original_hybrid_results(retrieve):
    rows = [candidate("old-hit", "设备读数核对。"), candidate("new-pool", "调整前后对照结果。")]
    provider = lambda texts: vectors(0.8, 0.49)
    original = retrieve("仪器偏差纠正", rows, embedding=provider)
    def broken(query, passages):
        raise RuntimeError("PRIVATE query and F:/secret/location")
    with pytest.warns(RuntimeWarning, match="重排.*退回") as emitted:
        result = retrieve("仪器偏差纠正", rows, embedding=provider, reranker=broken)
    assert [row["reference_id"] for row in result] == [row["reference_id"] for row in original]
    assert [row["reference_id"] for row in result] == ["old-hit"]
    assert result[0]["reranker_status"] == "error:RuntimeError"
    assert "PRIVATE" not in str(result) + str(emitted[0].message)
    assert "F:/secret" not in str(result) + str(emitted[0].message)


def test_partial_reranker_iterator_failure_does_not_escape_fallback(retrieve):
    def inconsistent(query, passages):
        yield 0.8
        raise RuntimeError("PRIVATE iterator failure")
    with pytest.warns(RuntimeWarning, match="重排.*退回"):
        result = retrieve("瓦斯", [candidate("gas", "瓦斯检查记录。")],
                          embedding=lambda texts: vectors(0.7), reranker=inconsistent)
    assert [row["reference_id"] for row in result] == ["gas"]
    assert result[0]["reranker_status"] == "invalid_scores"
    assert "PRIVATE" not in str(result)


@pytest.mark.parametrize("scores", [[], [0.5, 0.6], [math.nan], [math.inf],
                                    [True], ["0.9"], [-0.1], [1.1], None])
def test_invalid_reranker_scores_have_explicit_safe_fallback(retrieve, scores):
    with pytest.warns(RuntimeWarning, match="重排.*退回"):
        result = retrieve("瓦斯", [candidate("gas", "瓦斯检查记录。")],
                          embedding=lambda texts: vectors(0.7), reranker=lambda query, passages: scores)
    assert [row["reference_id"] for row in result] == ["gas"]
    assert result[0]["reranker_status"] == "invalid_scores"


def test_pool_is_bounded_deduplicated_and_has_stable_ties(retrieve):
    rows = [candidate(f"ref-{index:03}", f"合成内容编号{index}") for index in range(130)]
    calls = []
    def reranker(query, passages):
        calls.append(passages)
        return [0.9] * len(passages)
    def embedding(texts):
        return vectors(*([0.57] * (len(texts) - 1)))
    forward = retrieve("遇到读数偏差该如何处置", rows, embedding=embedding, reranker=reranker, limit=5)
    backward = retrieve("遇到读数偏差该如何处置", list(reversed(rows)), embedding=embedding, reranker=reranker, limit=5)
    assert forward == backward
    assert len(forward) == 5
    assert len(calls) == 2 and all(len(batch) <= 100 for batch in calls)
    assert all(len(set(batch)) == len(batch) for batch in calls)
    assert all(row["reranker_status"] == "available" for row in forward)


def test_empty_retrieval_does_not_call_reranker(retrieve):
    def forbidden(*args):
        raise AssertionError("empty retrieval invoked reranker")
    assert retrieve("", [candidate("a", "瓦斯")], reranker=forbidden) == []
    assert retrieve("瓦斯", [], reranker=forbidden) == []


def test_reranker_must_be_callable(retrieve):
    with pytest.raises(ValueError, match="reranker"):
        retrieve("瓦斯", [], reranker=object())


@pytest.mark.parametrize("embedding_failed", [False, True])
def test_missing_dense_vectors_label_the_actual_lexical_rank(retrieve, embedding_failed):
    def broken(texts):
        raise RuntimeError("synthetic unavailable")
    kwargs = {"reranker": lambda query, passages: [.9] * len(passages)}
    if embedding_failed:
        with pytest.warns(RuntimeWarning, match="向量.*词法"):
            result = retrieve("瓦斯", [candidate("gas", "瓦斯检查记录。")], embedding=broken, **kwargs)
    else:
        result = retrieve("瓦斯", [candidate("gas", "瓦斯检查记录。")], **kwargs)
    row = result[0]
    assert row["semantic_score"] is None and row["semantic_rank"] is None
    assert row["retrieval_rank_source"] == "lexical" and row["retrieval_rank"] == 1
    assert any("词法第1" in reason for reason in row["rerank_reasons"])
    assert not any("语义第" in reason for reason in row["rerank_reasons"])
