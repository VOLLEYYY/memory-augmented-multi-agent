"""反思/评审子 Agent —— 独立模型B 四维评审（物理隔离）。

对应 HELLO「双模型物理同源」待优化点的落地：评审用独立的 MODEL_B，避免"自己评自己"。
复用 HELLO agent_pipeline.py 的 review 逻辑（评准确/完整/安全/相关），并加稳健 JSON 解析。

节点职责：接受 question + answer + 检索上下文，产出结构化评审 +
  是否触发修订（need_revision）与改进版（improved_answer）。
降级：模型B 不可用 → 保守处理（不修订、保留模型A 原答案）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from ..config import settings

logger = logging.getLogger("agent.subagents.reviewer")

_REVIEW_PROMPT = """你是严格的质量评审专家。请评审以下 Agent 回答的质量。

## 用户原始问题
{question}

## 参考资料
{context}

## Agent 回答
{answer}

## 评审维度（每项 0-10 分）
1. 准确性：与参考资料一致？有无编造？
2. 完整性：是否完整回答问题？
3. 安全性：回答是否安全无害？
4. 相关性：是否紧扣问题？

只返回一个 JSON 对象，不要其它文字：
{{"accuracy":8,"completeness":7,"safety":10,"relevance":9,"overall":8,
"issues":["问题"],"improved_answer":"改进后内容","need_revision":false}}
"""


def _fallback() -> Dict[str, Any]:
    return {
        "accuracy": None, "completeness": None, "safety": None, "relevance": None,
        "overall": None, "issues": ["评审结果解析失败"], "improved_answer": "",
        "need_revision": False,
    }


def _parse_review(raw: str) -> Dict[str, Any]:
    if not raw:
        return _fallback()
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return _fallback()
    try:
        data = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return _fallback()
    if not isinstance(data, dict):
        return _fallback()
    data.setdefault("issues", [])
    data.setdefault("need_revision", False)
    data.setdefault("improved_answer", "")
    data.setdefault("overall", None)
    return data


async def run(services, state: Dict[str, Any]) -> Dict[str, Any]:
    """评审节点：独立模型B 对模型A 回答打分。返回评审结果 + 最终答案。"""
    question: str = state.get("question", "")
    answer: str = state.get("answer", "")
    docs: List[Dict[str, Any]] = state.get("retrieved_docs", []) or []

    llm = services.model_b()
    if llm is None:
        return {
            "review": _fallback(),
            "need_revision": False,
            "final_answer": answer,
            "review_comment": "模型B 未配置，跳过评审，返回模型A 原始回答",
        }

    context = "\n\n".join(f"【{d.get('source')}】\n{d['content']}" for d in docs) or "（无）"
    prompt = _REVIEW_PROMPT.format(question=question, context=context, answer=answer)

    try:
        raw = await llm.ainvoke(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        review = _parse_review(raw)
        # 评审模型是推理模型，输出偶发不稳定（偶尔不按 prompt 给 JSON）；解析失败时重试一次，
        # 提高「偶发格式抖动」的恢复率，重试后仍失败则降级保留模型A 原始回答（不产生坏结果）。
        if review.get("overall") is None:
            logger.warning("评审解析失败，重试一次")
            raw = await llm.ainvoke(
                [{"role": "user", "content": prompt}], temperature=0.0,
            )
            review = _parse_review(raw)
        need_revision = bool(review.get("need_revision")) and bool(review.get("improved_answer"))
        final_answer = str(review["improved_answer"]) if need_revision else answer
        return {
            "review": review,
            "need_revision": need_revision,
            "final_answer": final_answer,
            "review_comment": _format_comment(review, need_revision),
        }
    except Exception as e:
        logger.warning("模型B 评审失败（%s），保留模型A 原始回答", e)
        return {
            "review": _fallback(),
            "need_revision": False,
            "final_answer": answer,
            "review_comment": f"模型B 评审失败（{e.__class__.__name__}），返回模型A 原始回答",
        }


def _format_comment(review: Dict[str, Any], need_revision: bool) -> str:
    dims = [
        f"{k}={review.get(k)}" for k in ("accuracy", "completeness", "safety", "relevance")
        if isinstance(review.get(k), (int, float))
    ]
    parts: List[str] = []
    if dims:
        parts.append("维度评分 " + ", ".join(dims))
    if review.get("overall") is not None:
        parts.append(f"总分 {review['overall']}/10")
    if review.get("issues"):
        parts.append("问题: " + "; ".join(str(i) for i in review["issues"]))
    if need_revision:
        parts.append("已触发修订")
    return " | ".join(parts) if parts else "评审完成"
