"""Operational topic inference keeps object classes and ambiguous supply apart."""
from importlib import import_module
import math

import pytest


@pytest.fixture
def retrieve():
    return import_module("app.services.knowledge.retrieval").retrieve


def candidate(reference_id, text):
    return {"reference_id": reference_id, "title": "合成要求", "text": text,
            "locator": {"paragraph": 1}, "version": "v1"}


def weak(texts):
    return [[1, 0], *[[.2, math.sqrt(.96)] for _ in texts[1:]]]


@pytest.mark.parametrize("question,text,topic", [
    ("迎头送风中断后应如何核验", "通风中断后核对风机状态。", "通风"),
    ("作业区新鲜风的供给停了怎么办", "主要通风机停运后检查现场。", "通风"),
    ("探头调整后要保存哪些比对信息", "传感器调校完成后保存测试记录。", "调校"),
    ("还原时应核对完整性和可恢复时点吗", "数据库备份恢复须进行恢复验证。", "备份恢复"),
    ("这个测值是谁何时因何修改的", "监测原始记录更正应记录操作者和变更理由。", "记录更正"),
    ("钻孔中的气体抽取是否达标应核对什么", "瓦斯抽采达标评判应核对相关指标。", "抽采"),
])
def test_compositional_query_recalls_its_formal_topic(retrieve, question, text, topic):
    result = retrieve(question, [candidate("support", text)], embedding=weak)
    assert [row["reference_id"] for row in result] == ["support"]
    assert topic in result[0]["matched_concepts"]
    assert topic in result[0]["inferred_query_topics"]


def test_unspecified_energy_keeps_parallel_supply_branches(retrieve):
    rows = [candidate("power", "恢复送电须由授权人员人工执行。"),
            candidate("gas", "恢复供气前核查专用阀门及管路压力。"),
            candidate("air", "恢复压风供给前确认压风管路状况。")]
    result = retrieve("恢复能源供给前需要核对什么材料", rows, embedding=weak)
    assert {row["reference_id"] for row in result} == {"power", "gas", "air"}
    assert all("能源恢复" in row["matched_concepts"] for row in result)
    assert all(row["topic_ambiguities"] == ["energy_supply_type_unspecified"] for row in result)
    assert all("恢复供电" not in row["matched_concepts"] for row in result)


@pytest.mark.parametrize("question,text", [
    ("调整探头吊挂位置和高度后怎么固定", "甲烷传感器调校后填写调校记录。"),
    ("探头到货后如何比对型号信息", "甲烷传感器调校后填写调校记录。"),
    ("数学试卷的答案怎么更正", "监测原始记录的错误更正保留新修订。"),
    ("如何还原手机出厂设置", "数据库备份恢复前进行恢复验证。"),
    ("厨房气体抽取是否达标", "煤层瓦斯抽采达标应核对参数。"),
    ("备用机组保养时怎样添加润滑油", "监控主机采用双机热备，备用主机自动投入。"),
    ("电源适配器接通后为什么发热", "恢复供电前应检查，由授权人员人工合闸。"),
    ("火灾事故调查会议结束，大家从哪里离开会议室", "事故中应按避灾路线撤离。"),
    ("仓库补来的配件如何验收", "补传数据保留原采样时间。"),
    ("公司新修订的年假管理办法何时执行", "监测原始记录的错误更正保留新修订。"),
    ("历史课上怎样评价风流人物", "通风异常应核对风流路线。"),
    ("我要给小说人物安排一个脱险结局", "事故中应按避灾路线撤离。"),
    ("主要通风机停运怎么处置", "局部通风机停运后及时检查。"),
    ("局部通风机停止运转如何处置", "主要通风机停运后及时检查。"),
    ("恢复供电应检查哪些条件", "恢复供气前检查专用管路。"),
    ("恢复供气应检查哪些条件", "恢复送电前检查联锁开关。"),
])
def test_different_objects_do_not_become_the_requested_topic(retrieve, question, text):
    assert retrieve(question, [candidate("other-topic", text)], embedding=weak) == []


def test_exact_locator_survives_a_different_device_topic(retrieve):
    row = {**candidate("exact", "主要通风机检查记录。"), "title": "AQ 1029-2019", "locator": {"clause": "6.3.1"}}
    result = retrieve("AQ1029-2019 第6.3.1条 局部通风机", [row], embedding=weak)
    assert [item["reference_id"] for item in result] == ["exact"]
