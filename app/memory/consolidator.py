"""记忆固化子 Agent —— 摘要压缩 + 去重 + 矛盾合并（核心卖点，面试重点讲）。

职责：把「近期情景记忆（episodic）」压缩提炼为「语义记忆（semantic）」，
防止记忆变成垃圾抽屉 —— 这是业界原话：semantic memory 不做 curation 就是 junk drawer。

流程：
  1. 拉取近期 episodic 记忆（按时间序）。
  2. 简单分桶（编码字节相似度 > 阈值 视为同话题，去重）。
  3. 每桶生成摘要：LLM 可用则提炼；不可用则降级为「保留最新 + 拼接」。
  4. 将摘要以 semantic 记忆落库（key=summary），实现「跨会话长期记忆」。
  5. 返回 ConsolidationResult（保留/合并/丢弃计数），供评测与前端展示。

与 graph 的关系：作为"后台子 Agent"异步触发（LangGraph 可在会话间隙调度），
不阻塞主问答链路 —— 这是多智能体「事件驱动」的一种落地。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..config import settings
from ..core.llm import LLMClient
from .episodic import EpisodicMemory, get_episodic_memory
from .models import ConsolidationResult, MemoryItem, MemoryType
from .semantic import SemanticMemory, get_semantic_memory

logger = logging.getLogger("agent.memory.consolidator")

# 分桶去重的相似度阈值（编码层字节重叠）
_DEDUP_THRESHOLD = 0.8
# 一次固化最多处理的记忆条数
_MAX_BATCH = 20

_SUMMARY_PROMPT = """你是一个记忆整理专家。请把下面几条「对话记忆」压缩成一条语义记忆。

对话记忆：
{memories}

请提取其中值得长期记住的用户事实、偏好、关键结论，合并成一句简洁的中文摘要。
如果存在矛盾，以最新为准。只输出一句话摘要，不要其它文字。
"""


class Consolidator:
    def __init__(
        self,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
    ):
        self.episodic = episodic or get_episodic_memory()
        self.semantic = semantic or get_semantic_memory()
        self._llm: LLMClient | None = None

    def _get_llm(self) -> LLMClient | None:
        if self._llm is None:
            try:
                self._llm = LLMClient(role="a")
            except Exception:
                self._llm = None
        return self._llm

    # ---- 简单分桶去重（同一话题合并）----
    def _bucketize(self, items: List[MemoryItem]) -> List[List[MemoryItem]]:
        buckets: List[List[MemoryItem]] = []
        for it in items:
            placed = False
            for bucket in buckets:
                if self._similar(it.content, bucket[0].content) >= _DEDUP_THRESHOLD:
                    bucket.append(it)
                    placed = True
                    break
            if not placed:
                buckets.append([it])
        return buckets

    @staticmethod
    def _similar(a: str, b: str) -> float:
        sa, sb = set(a.lower()), set(b.lower())
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        return inter / max(len(sa), len(sb))

    async def _summarize_one(self, memories: List[MemoryItem]) -> str:
        """对一桶记忆生成摘要：优先 LLM，失败降级为规则拼接。"""
        text = "\n".join(f"- {it.content}" for it in memories)
        llm = self._get_llm()
        if llm is not None:
            try:
                prompt = _SUMMARY_PROMPT.format(memories=text)
                summary = await llm.ainvoke([{"role": "user", "content": prompt}])
                if summary and summary.strip():
                    return summary.strip()
            except Exception as e:
                logger.warning("固化摘要 LLM 失败（%s），降级为规则拼接", e)
        # 降级：保留最新一条 + 合并去重的关键词
        latest = memories[-1].content
        return latest if len(memories) == 1 else f"{latest}（合并自 {len(memories)} 条相关记忆）"

    async def consolidate(self, overrides: List[MemoryItem] | None = None) -> ConsolidationResult:
        """执行一次固化。overrides：显式传入要固化的记忆（默认从 episodic 拉近期记忆）。"""
        if overrides is not None:
            episodes = overrides
        else:
            # 固化需要「近期发生了什么」，按时间序拉取，而非按相似度 search（search 会被
            # 「语义相近但时间无关」的记忆干扰）。
            episodes = await self.episodic.list_recent(_MAX_BATCH)

        if not episodes:
            return ConsolidationResult(summary="")

        buckets = self._bucketize(episodes[:_MAX_BATCH])
        result = ConsolidationResult(
            merged_count=max(0, len(episodes) - len(buckets)),
            dropped_count=0,
        )

        for bucket in buckets:
            summary = await self._summarize_one(bucket)
            if summary:
                # 落一条语义记忆（key=summary），作为跨会话长期记忆
                item = await self.semantic.upsert(
                    key="user.summary", value=summary, source="consolidator",
                    importance=0.6, content=summary,
                )
                result.kept_memories.append(item)
        result.summary = (
            f"固化完成：{len(buckets)} 组记忆，合并 {result.merged_count} 条，"
            f"保留 {len(result.kept_memories)} 条摘要"
        )
        return result


_instance: Consolidator | None = None


def get_consolidator() -> Consolidator:
    global _instance
    if _instance is None:
        _instance = Consolidator()
    return _instance
