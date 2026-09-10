"""评审稳健解析单测 —— 覆盖 `_parse_review` 对「脏输出」的鲁棒性（Day 10）。

背景：模型 B 输出评审 JSON 时，常带 Markdown 代码块包裹、前后多余文字、甚至漏字段。
`_parse_review` 要保证任何脏输入都「要么解析出正确字段，要么干净降级」，不抛异常、不产出坏结果。

零依赖：只测 `reviewer._parse_review` 纯函数，不碰模型 / qdrant / langgraph。
运行：cd multi-agent && pytest -q tests/test_reviewer.py
"""
from __future__ import annotations

import json


def _good() -> dict:
    return {
        "accuracy": 8, "completeness": 7, "safety": 10, "relevance": 9,
        "overall": 8, "issues": ["资料外编造"], "improved_answer": "", "need_revision": False,
    }


def test_parse_review_plain_json():
    from app.subagents.reviewer import _parse_review
    data = _parse_review(json.dumps(_good(), ensure_ascii=False))
    assert data["accuracy"] == 8
    assert data["overall"] == 8
    assert data["issues"] == ["资料外编造"]
    assert data["need_revision"] is False


def test_parse_review_markdown_fence():
    from app.subagents.reviewer import _parse_review
    raw = "```json\n" + json.dumps(_good(), ensure_ascii=False) + "\n```"
    data = _parse_review(raw)
    assert data["accuracy"] == 8
    assert data["overall"] == 8


def test_parse_review_with_surrounding_text():
    from app.subagents.reviewer import _parse_review
    raw = "好的，以下是评审结果：\n" + json.dumps(_good(), ensure_ascii=False) + "\n以上是评审。"
    data = _parse_review(raw)
    assert data["overall"] == 8
    assert data["issues"] == ["资料外编造"]


def test_parse_review_missing_fields_get_defaults():
    from app.subagents.reviewer import _parse_review
    # 模型漏了几个字段：issues / need_revision / improved_answer 应由 setdefault 兜底
    data = _parse_review('{"accuracy": 9, "overall": 9}')
    assert data["accuracy"] == 9
    assert data["overall"] == 9
    assert data["issues"] == []
    assert data["need_revision"] is False
    assert data["improved_answer"] == ""


def test_parse_review_invalid_json_falls_back():
    from app.subagents.reviewer import _parse_review
    data = _parse_review("这不是 JSON，是一段纯文字")
    assert data["overall"] is None
    assert data["need_revision"] is False
    assert "解析失败" in data["issues"][0]


def test_parse_review_non_dict_falls_back():
    from app.subagents.reviewer import _parse_review
    # JSON 数组不是 dict，应降级而非崩
    data = _parse_review("[1, 2, 3]")
    assert data["overall"] is None
    assert data["need_revision"] is False


def test_parse_review_empty_and_none_fall_back():
    from app.subagents.reviewer import _parse_review
    assert _parse_review("")["overall"] is None
    assert _parse_review(None)["overall"] is None
