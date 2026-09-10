"""记忆子 Agent —— 读取/写入跨会话长期记忆。

两个函数（各作为 LangGraph 的一个节点/工具）：
  1. `run(services, state)`  —— 读取：用当前问题检索 episodic + semantic 记忆，
     拼成 memory_context 注入主控上下文（跨会话记忆生效的关键一步）。
  2. `record(services, state)` —— 写入：把本轮问答作为一条 episodic 记忆落库，
     供后台固化子 Agent 提炼为长期语义记忆。

设计要点（痛点④ + 记忆一致性）：
- 多个子 Agent 共享同一个 MemoryStore（services.episodic/.semantic），
  谁写都能被其它 Agent 读到 —— 这就是「统一记忆神经中枢」，避免各 Agent 各说各话。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ..memory.models import MemoryItem, MemoryType
from ..memory.working import WorkingMemory

logger = logging.getLogger("agent.subagents.memory")


async def run(services, state: Dict[str, Any]) -> Dict[str, Any]:
    """读取记忆：工作记忆（最近对话）+ 长期记忆（episodic + semantic）→ memory_context。"""
    question: str = state.get("question", "")

    parts = []
    hit = False
    # 工作记忆：checkpointer 累积的最近对话（含上一轮 Q:A），截断后注入
    messages = state.get("messages", [])
    try:
        wm = WorkingMemory().summarize(messages)
        if wm:
            parts.append(f"[工作记忆·最近对话]\n{wm}")
    except Exception as e:
        logger.warning("工作记忆摘要失败（%s）：%s", e.__class__.__name__, e)
    # 长期记忆：episodic + semantic
    # 跨用户隔离（P0 修复）：episodic 检索带上 thread_id，只在当前用户的记忆范围内召回，
    # 避免 user-2 命中 user-1 的记忆（隐私泄露 + 回答串线）。
    thread_id = state.get("thread_id", "") or None
    try:
        epi = await services.episodic.search(question, thread_id=thread_id)
        if epi.hit:
            parts.append(epi.as_content())
            hit = True
    except Exception as e:
        logger.warning("情景记忆检索失败（%s）：%s", e.__class__.__name__, e)
    try:
        sem = await services.semantic.search(question)
        if sem.hit:
            parts.append(sem.as_content())
            hit = True
    except Exception as e:
        logger.warning("语义记忆检索失败（%s）：%s", e.__class__.__name__, e)

    return {"memory_context": "\n".join(parts), "memory_hit": hit}


async def record(services, state: Dict[str, Any]) -> Dict[str, Any]:
    """写入本轮问答为一条情景记忆（供固化），并把本轮 Q/A 追加进工作记忆。"""
    question: str = state.get("question", "")
    if not question:
        return {}
    answer = state.get("final_answer") or state.get("answer", "")
    content = f"用户问：{question}"
    if answer:
        content += f"\n助答：{str(answer)[:300]}"
    try:
        item = MemoryItem(
            id=f"epi_{int(time.time())}_{abs(hash(question)) % 1000}",
            type=MemoryType.EPISODIC,
            content=content,
            importance=0.4,
            source="memory_agent",
            # thread_id 落 metadata，供检索端做跨用户隔离（P0 修复）
            metadata={"question": question, "thread_id": state.get("thread_id", "")},
        )
        await services.episodic.add(item)
        # 追加本轮回答到工作记忆（checkpointer 用 Annotated[operator.add] 跨轮累积）
        msgs = [{"role": "assistant", "content": str(answer)[:300]}] if answer else []
        return {"memory_written": True, "messages": msgs}
    except Exception as e:
        logger.warning("情景记忆写入失败（%s）：%s", e.__class__.__name__, e)
        return {"memory_written": False}
