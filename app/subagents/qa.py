"""知识问答子 Agent —— 混合检索 + 上下文增强生成。

节点职责（在 LangGraph 中作为 one node）：
  输入：state 里的 question + memory_context（来自记忆子 Agent）
  处理：向 retriever 检索相关资料 → 组装「知识库 + 记忆」增强提示 → 模型A 生成
  返回：answer / sources / retrieved_docs

降级链：模型中任一环节失败 → 返回「检索到的资料片段」作为回答（不阻塞主流程）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..config import settings
from ..core.llm import LLMClient

logger = logging.getLogger("agent.subagents.qa")

_AUGMENT_PROMPT = """你是知识库问答助手。请严格基于以下参考资料回答用户问题。

## 参考资料（知识库检索结果）
{context}

## 相关长期记忆（用户偏好/历史，仅作补充，勿编造）
{memory}

## 用户问题
{question}

## 要求
1. 严格基于参考资料回答，不要编造。
2. 若资料中没有，回答"根据现有资料无法回答"。
3. 引用时用【来源】标注出处。
"""


def _build_context(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "（无参考资料）"
    return "\n\n".join(
        f"【来源：{d.get('source', '未知')}】\n{d['content']}" for d in docs
    )


def _build_messages(question: str, docs: List[Dict[str, Any]], memory: str) -> List[Dict[str, str]]:
    prompt = _AUGMENT_PROMPT.format(
        context=_build_context(docs), memory=memory or "（无）", question=question,
    )
    return [{"role": "user", "content": prompt}]


async def run(services, state: Dict[str, Any]) -> Dict[str, Any]:
    """问答节点：检索 → 生成。返回部分状态更新。"""
    question: str = state.get("question", "")
    memory: str = state.get("memory_context", "")

    # ---- Step 1: 混合检索 ----
    retriever = await services.ensure_retriever()
    try:
        docs = await retriever.search(question, settings.top_k)
    except Exception as e:
        logger.warning("检索失败（%s），用空上下文", e)
        docs = []

    # ---- Step 2: 模型A 生成 ----
    answer = ""
    llm: LLMClient | None = services.model_a()
    if llm is not None:
        try:
            answer = await llm.ainvoke(_build_messages(question, docs, memory))
        except Exception as e:
            logger.warning("模型A 生成失败（%s），降级为资料片段", e)
    if not answer or not str(answer).strip():
        snippet = docs[0]["content"] if docs else "（模型未配置 / 无检索结果）"
        answer = f"[降级] 模型不可用，以下为检索到的相关资料片段：\n{snippet}"

    return {
        "answer": str(answer),
        "sources": [d.get("source") for d in docs],
        "retrieved_docs": docs,
    }
