"""Day 13 单测 —— 投诉→转人工路由（Day 12 评测暴露的缺口）+ 意图分流优先级。

零重依赖：不碰 qdrant / sentence-transformers / torch / 真实模型。
运行：cd multi-agent && pytest -q tests/test_day13.py
"""
from __future__ import annotations


def test_resolve_escalation_complaint():
    """投诉/差评/索赔/找人工 → 触发转人工。"""
    from app.subagents.tools import resolve_escalation

    assert resolve_escalation("我要投诉你们的客服态度差") is True
    assert resolve_escalation("对处理结果不满意，我要投诉升级") is True
    assert resolve_escalation("产品质量有严重问题，我要投诉并索赔") is True
    assert resolve_escalation("帮我转人工") is True


def test_resolve_escalation_not_triggered():
    """普通咨询 / 工具意图不触发转人工。"""
    from app.subagents.tools import resolve_escalation

    assert resolve_escalation("门锁保修多久") is False
    assert resolve_escalation("我要申请退款，订单号 A123") is False
    assert resolve_escalation("查一下我的订单 A123") is False
    # 「退货政策咨询」含「退货」但不含投诉诉求 → 不转人工（仍走工具/QA，这是已知关键词路由边界）
    assert resolve_escalation("七天无理由退货是什么政策？") is False


def test_format_escalation_mentions_handoff():
    """转人工话术包含人工客服 + 电话，依据售后服务流程的「投诉升级」条款。"""
    from app.subagents.tools import format_escalation

    text = format_escalation()
    assert ("转人工" in text) or ("人工客服" in text)
    assert "400-800-1234" in text


def test_route_escalates_before_tools():
    """graph._route 分流顺序：投诉（转人工）优先于工具，工具优先于 QA。"""
    from app.core.graph import _route

    assert _route({"question": "我要投诉你们的客服态度差"}) == "escalate"
    # 同时含「投诉」和「订单」关键词 → 投诉优先转人工
    assert _route({"question": "我要投诉，订单 A123 有问题"}) == "escalate"
    assert _route({"question": "查一下我的订单 A123"}) == "tools"
    assert _route({"question": "门锁保修多久"}) == "qa"
