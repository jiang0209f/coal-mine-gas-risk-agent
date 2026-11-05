"""Pure retrieval behavior, using synthetic passages and injected local vectors."""
from copy import deepcopy
from importlib import import_module
import math

import pytest


@pytest.fixture
def retrieve():
    return import_module("app.services.knowledge.retrieval").retrieve


def candidate(reference_id, text, title="合成作业规程", locator=None, version="v1"):
    return {"reference_id": reference_id, "title": title, "text": text,
            "locator": locator or {"page": 1}, "version": version}


def test_compound_query_does_not_recall_table_header_with_only_generic_suffix(retrieve):
    rows = [candidate("header", "合成参数 数值", title="合成资料"), candidate("row", "瓦斯参数 合成值", title="合成资料")]
    assert [row["reference_id"] for row in retrieve("瓦斯参数", rows)] == ["row"]


def test_human_readable_source_is_searchable_and_terms_follow_query_order(retrieve):
    row = {**candidate("row", "核对记录。", title="瓦斯监测规程"), "source": "合成离线测点文件"}
    result = retrieve("瓦斯 合成", [row])
    assert result[0]["matched_terms"] == ["瓦斯", "合成"]


def test_exact_body_phrase_does_not_recall_sibling_only_because_of_document_title(retrieve):
    rows = [candidate("evidence", "合成闭环依据：核对记录。", title="合成闭环验收资料"),
            candidate("other", "合成复核记录：登记签字。", title="合成闭环验收资料")]
    assert [row["reference_id"] for row in retrieve("合成闭环依据", rows)] == ["evidence"]


def test_partial_title_prefix_without_the_requested_topic_is_not_evidence(retrieve):
    rows = [candidate("evidence", "合成闭环依据：核对记录。", title="合成闭环验收资料")]
    assert retrieve("合成闭环经验", rows) == []


@pytest.mark.parametrize("question,text", [
    ("啥时候可以重新送电？", "恢复供电须完成现场检查。"),
    ("探头如何标校", "甲烷传感器调校应核对记录。"),
    ("什么情况下要撤人", "异常情况下组织人员撤离。"),
    ("抽放效果怎样核查", "抽采效果应按原始资料核查。"),
    ("甲烷浓度超标怎么处理", "瓦斯超限后应核查原因。"),
    ("gas detector calibration", "甲烷传感器调校记录。"),
])
def test_synonym_question_recalls_the_relevant_passage(retrieve, question, text):
    rows = [candidate("unrelated", "财务票据按月归档。"), candidate("relevant", text)]
    result = retrieve(question, rows)
    assert [item["reference_id"] for item in result] == ["relevant"]
    assert result[0]["lexical_score"] > 0
    assert result[0]["semantic_score"] is None
    assert result[0]["rerank_reasons"]


@pytest.mark.parametrize("question,text", [
    ("读数正常后能让电源自己接通吗", "恢复供电前应完成检查，由授权人员人工合闸。"),
    ("主电脑死机了，备用机多快接手", "双机热备运行时，备用主机应按规定自动投入。"),
    ("探头用气样标好了，要留什么单子", "甲烷传感器调校结束后填写调校记录。"),
    ("断开重连后补来的值能当当前值吗", "补传数据应保留原采样时间，并登记接收时间。"),
    ("那份拷贝真能还原吗", "备份完整性检查后，应进行恢复验证。"),
    ("录入错了能覆盖旧数吗", "错误更正应保留原始记录并登记新修订。"),
    ("大风机停了怎么处置", "主要通风机停止后应按适用条文处置。"),
])
def test_operational_paraphrases_recall_formal_topics(retrieve, question, text):
    result = retrieve(question, [candidate("evidence", text), candidate("other", "财务票据定期归档。")])
    assert [row["reference_id"] for row in result] == ["evidence"]
    assert result[0]["matched_concepts"]


@pytest.mark.parametrize("question,text", [
    ("会议结束后从哪里离开", "事故后应按避灾路线撤离。"),
    ("火灾调查记录由谁签字", "发生事故后应按避灾路线撤离。"),
    ("电源适配器在哪里买", "恢复供电前应完成检查。"),
])
def test_colloquial_expansion_requires_the_action_context(retrieve, question, text):
    assert retrieve(question, [candidate("other-topic", text, title="安全要求")]) == []


@pytest.mark.parametrize("question,text,similarity", [
    ("历史课上怎样评价风流人物", "通风系统异常时核对风流路线。", 0.379),
    ("如何还原手机出厂设置", "数据库备份恢复前进行恢复验证。", 0.436),
    ("我要给小说人物安排一个脱险结局", "事故中应按避灾路线撤离。", 0.440),
])
def test_polysemous_words_need_a_mining_or_data_context(retrieve, question, text, similarity):
    rows = [candidate("other-domain", text, title="安全要求")]
    vectors = [[1, 0], [similarity, math.sqrt(1 - similarity ** 2)]]
    assert retrieve(question, rows, embedding=lambda texts: vectors) == []


@pytest.mark.parametrize("question,text", [
    ("备用机组保养时怎样添加润滑油", "监控主机采用双机热备，备用主机自动投入。"),
    ("数学试卷的答案怎么更正", "监测原始记录的错误更正保留新修订。"),
    ("电源适配器接通后为什么发热", "恢复供电前应检查，由授权人员人工合闸。"),
    ("火灾事故调查会议结束，大家从哪里离开会议室", "事故中应按避灾路线撤离。"),
    ("仓库补来的配件如何验收", "补传数据保留原采样时间。"),
    ("公司新修订的年假管理办法何时执行", "监测原始记录的错误更正保留新修订。"),
])
@pytest.mark.parametrize("with_vectors", [False, True])
def test_operational_aliases_require_their_device_or_data_object(retrieve, question, text, with_vectors):
    embedding = (lambda texts: [[1, 0], [0.2, math.sqrt(0.96)]]) if with_vectors else None
    assert retrieve(question, [candidate("other-topic", text, title="安全要求")], embedding=embedding) == []


def test_complete_standard_number_beats_similar_title_and_other_edition(retrieve):
    rows = [candidate("wrong-edition", "甲烷传感器使用要求。", title="AQ 1029-2007"),
            candidate("exact", "监控设备核对。", title="AQ 1029-2019"),
            candidate("different", "甲烷传感器使用要求。", title="AQ 6201-2019")]
    result = retrieve("AQ1029—2019 中的传感器要求", rows)
    assert result[0]["reference_id"] == "exact"
    assert "AQ1029-2019" in result[0]["exact_matches"]
    assert all(item["reference_id"] != "wrong-edition" for item in result)


def test_article_numbers_normalize_chinese_and_arabic_without_partial_match(retrieve):
    rows = [candidate("eleven", "第十一条：恢复供电条件。"),
            candidate("seventy-one", "第71条：现场核对。"),
            candidate("two-eleven", "第二百一十一条：恢复供电条件。")]
    result = retrieve("请查第七十一条", rows)
    assert [item["reference_id"] for item in result] == ["seventy-one"]
    assert "第71条" in result[0]["exact_matches"]


def test_section_numbers_respect_boundaries_and_structured_locators(retrieve):
    rows = [candidate("wrong", "传感器要求。", title="AQ1029-2019", locator={"clause": "16.3.1"}),
            candidate("exact", "记录核对。", title="AQ1029-2019", locator={"clause": "6.3.1"})]
    result = retrieve("AQ 1029-2019 第6.3.1条", rows)
    assert result[0]["reference_id"] == "exact"
    assert "6.3.1" in result[0]["exact_matches"]


@pytest.mark.parametrize("question", ["今天北京的天气怎么样", "矿井员工工资和社保怎么计算", "瓦斯矿井员工工资和社保怎么计算"])
def test_unrelated_question_does_not_return_domain_passages(retrieve, question):
    assert retrieve(question, [candidate("gas", "瓦斯超限后的断电、撤离和复电条件。")]) == []


def test_injected_vectors_recall_synonyms_and_rerank_without_network(retrieve):
    calls = []
    def embedding(texts):
        calls.append(texts)
        return [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [1.0, 0.0]]

    rows = [candidate("orthogonal", "财务票据归档。"),
            candidate("related", "维护人员的检测安排。"),
            candidate("best", "设备经过调整并记录比对结果。")]
    result = retrieve("仪器读数需要纠偏", rows, embedding=embedding)
    assert [item["reference_id"] for item in result] == ["best", "related"]
    assert result[0]["semantic_score"] == pytest.approx(1.0)
    assert result[1]["semantic_score"] == pytest.approx(0.8)
    assert result[0]["lexical_score"] == 0
    assert result[0]["retrieval_mode"] == "hybrid"
    assert result[0]["embedding_status"] == "available"
    assert len(calls) == 1 and len(calls[0]) == len(rows) + 1
    assert calls[0][0] == "仪器读数需要纠偏"


def test_negative_or_orthogonal_vectors_do_not_force_irrelevant_recall(retrieve):
    result = retrieve("明天是否降雨", [candidate("gas", "瓦斯超限断电。"), candidate("wind", "通风检查。")],
                      embedding=lambda texts: [[1, 0], [0, 1], [-1, 0]])
    assert result == []


def test_colloquial_escape_question_recalls_route_without_shared_terms(retrieve):
    # Measured local embedding score for this missed synthetic question/passage.
    similarity = 0.5786556042
    rows = [candidate("escape-route", "发生事故后应按避灾路线撤离危险区域，核对出口与安全路线。",
                      title="事故处置")]
    result = retrieve("遇险时人员从哪里离开", rows,
                      embedding=lambda texts: [[1, 0], [similarity, math.sqrt(1 - similarity ** 2)]])
    assert [row["reference_id"] for row in result] == ["escape-route"]
    assert result[0]["semantic_score"] == pytest.approx(similarity)
    assert result[0]["matched_terms"] == []


@pytest.mark.parametrize("text,similarity", [
    ("旅客离开列车时应寻找车厢出口。", 0.5338),
    ("游客游览结束后从景区出口离开。", 0.5361),
])
def test_similar_cross_topic_departure_does_not_recall_an_escape_rule(retrieve, text, similarity):
    result = retrieve("遇险时人员从哪里离开", [candidate("other-domain", text, title="出行服务")],
                      embedding=lambda texts: [[1, 0], [similarity, math.sqrt(1 - similarity ** 2)]])
    assert result == []


def test_semantic_answer_outranks_generic_action_word_collisions(retrieve):
    rows = [candidate("action", "通风不良时停止全部电路，撤出作业人员，向调度室报告。", title="异常处置"),
            candidate("statistics", "风路停电汇报台账只用于统计巡检次数。", title="台账整理")]
    similarities = (0.6724, 0.50)
    vectors = [[1, 0], *[[score, math.sqrt(1 - score ** 2)] for score in similarities]]
    result = retrieve("风路堵住不畅通了，要不要先停电，向谁汇报？", rows, embedding=lambda texts: vectors)
    assert [row["reference_id"] for row in result] == ["action", "statistics"]
    assert result[0]["semantic_score"] > result[1]["semantic_score"]
    assert result[0]["lexical_score"] < result[1]["lexical_score"]


def test_exact_standard_priority_survives_a_higher_unrelated_vector_score(retrieve):
    rows = [candidate("target", "设备核查。", title="GB 41022-2021"),
            candidate("wrong", "设备核查。", title="GB 40880-2021")]
    result = retrieve("GB41022-2021", rows, embedding=lambda texts: [[1, 0], [0, 1], [1, 0]])
    assert [item["reference_id"] for item in result] == ["target"]


def test_lexical_and_vector_scores_remain_finite_and_bounded(retrieve):
    result = retrieve("瓦斯超限", [candidate("a", "瓦斯超限记录。")],
                      embedding=lambda texts: [[1e300, 1e300], [1e300, 1e300]])
    for key in ("score", "lexical_score", "semantic_score"):
        assert math.isfinite(result[0][key])
        assert 0 <= result[0][key] <= 1


@pytest.mark.parametrize("vectors", [
    [], [[1, 0]], [[1, 0], [1, 0, 1]], [[1, 0], [math.nan, 0]],
    [[1, 0], [math.inf, 0]], [[1, 0], [0, 0]], [[1, 0], [True, 0]],
])
def test_invalid_vector_batch_has_explicit_lexical_fallback(retrieve, vectors):
    with pytest.warns(RuntimeWarning, match="向量.*词法"):
        result = retrieve("瓦斯", [candidate("a", "瓦斯检查记录。")], embedding=lambda texts: vectors)
    assert result[0]["reference_id"] == "a"
    assert result[0]["retrieval_mode"] == "lexical_fallback"
    assert result[0]["embedding_status"] == "invalid_vectors"
    assert result[0]["semantic_score"] is None


def test_embedding_failure_is_observable_even_when_no_passage_matches(retrieve):
    def unavailable(texts):
        raise RuntimeError("synthetic unavailable")
    with pytest.warns(RuntimeWarning, match="向量.*词法"):
        result = retrieve("北京天气", [candidate("a", "瓦斯检查记录。")], embedding=unavailable)
    assert result == []


def test_english_aliases_do_not_match_inside_unrelated_words(retrieve):
    assert retrieve("vegas hotel", [candidate("gas", "瓦斯检查记录。")]) == []


def test_runtime_failure_retains_lexical_results_and_a_safe_error_code(retrieve):
    def unavailable(texts):
        raise RuntimeError("synthetic private content must not enter ranking metadata")
    with pytest.warns(RuntimeWarning, match="向量.*词法"):
        result = retrieve("瓦斯", [candidate("a", "瓦斯检查记录。")], embedding=unavailable)
    assert result[0]["reference_id"] == "a"
    assert result[0]["embedding_status"] == "error:RuntimeError"
    assert result[0]["retrieval_mode"] == "lexical_fallback"
    assert "private content" not in str(result)


def test_low_coverage_word_collision_needs_vector_support(retrieve):
    rows = [candidate("wage-rule", "项目建设期间不得拖欠工资。")]
    result = retrieve("这个项目的工资怎么发", rows,
                      embedding=lambda texts: [[1, 0], [0.5, math.sqrt(0.75)]])
    assert result == []


def test_complete_long_phrase_survives_a_low_vector_score(retrieve):
    rows = [candidate("grounding", "接地装置应核对原始记录。")]
    result = retrieve("接地装置检查方法", rows, embedding=lambda texts: [[1, 0], [0, 1]])
    assert [row["reference_id"] for row in result] == ["grounding"]


def test_ties_are_stable_limits_apply_and_inputs_are_not_mutated(retrieve):
    rows = [candidate("c", "瓦斯记录。"), candidate("a", "瓦斯记录。"), candidate("b", "瓦斯记录。")]
    original = deepcopy(rows)
    forward = retrieve("瓦斯", rows, limit=2)
    backward = retrieve("瓦斯", list(reversed(rows)), limit=2)
    assert [item["reference_id"] for item in forward] == ["a", "b"]
    assert forward == backward and rows == original
    assert forward[0]["locator"] == {"page": 1} and forward[0]["version"] == "v1"


@pytest.mark.parametrize("query", [None, 42, ["瓦斯"]])
def test_query_must_be_text(retrieve, query):
    with pytest.raises(ValueError, match="检索词"):
        retrieve(query, [])


@pytest.mark.parametrize("limit", [0, -1, 51, True, 1.5, "5"])
def test_limit_must_be_a_bounded_integer(retrieve, limit):
    with pytest.raises(ValueError, match="limit"):
        retrieve("瓦斯", [], limit=limit)


@pytest.mark.parametrize("rows", [None, "瓦斯", [{}], [candidate("a", "")],
    [candidate("a", "瓦斯"), candidate("a", "重复编号")]])
def test_candidates_need_valid_unique_references_and_content(retrieve, rows):
    with pytest.raises(ValueError, match="候选"):
        retrieve("瓦斯", rows)


def test_empty_query_or_candidate_list_does_not_invoke_embedding(retrieve):
    def forbidden(texts):
        raise AssertionError("empty retrieval must not invoke embeddings")
    assert retrieve("  ", [candidate("a", "瓦斯")], embedding=forbidden) == []
    assert retrieve("瓦斯", [], embedding=forbidden) == []


def test_noncallable_embedding_is_a_configuration_error(retrieve):
    with pytest.raises(ValueError, match="embedding"):
        retrieve("瓦斯", [], embedding=object())
