"""Day 8 工作记忆接入验证 —— memory_agent 读工作记忆 + record 追加 + reducer 跨轮累积。

说明：跨进程「重启」的 sqlite 持久化由 AsyncSqliteSaver 承担（Day1 起服务已验 memory.db 落盘）；
这里用 InMemorySaver 验证「messages 跨轮累积 + 工作记忆注入」这条逻辑链路（不碰 sqlite/aiosqlite，
避免 Windows 短命进程反复 kill 导致的 aiosqlite worker 锁问题）。

前两条零依赖（直接调 memory_agent 纯函数）；第三条依赖 langgraph（项目核心依赖）。
运行：cd multi-agent && pytest -q tests/test_working_memory.py
"""
from __future__ import annotations

import os

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.core.graph import AgentState
from app.memory.episodic import _InMemoryEpisodic
from app.memory.semantic import SemanticMemory
from app.subagents import memory_agent


class _FakeServices:
    def __init__(self, tmp: str):
        self.episodic = _InMemoryEpisodic()
        self.semantic = SemanticMemory(path=os.path.join(tmp, "sem.db"))
        self.model_a = lambda: None
        self.model_b = lambda: None


# ---- memory_agent.run：读工作记忆（checkpointer 累积的最近对话）注入 memory_context ----
@pytest.mark.asyncio
async def test_memory_agent_injects_working_memory(tmp_path):
    services = _FakeServices(str(tmp_path))
    state = {
        "question": "门锁想换货",
        "messages": [
            {"role": "user", "content": "门锁保修多久"},
            {"role": "assistant", "content": "门锁保修 2 年"},
            {"role": "user", "content": "门锁想换货"},
        ],
    }
    out = await memory_agent.run(services, state)
    assert "工作记忆" in out["memory_context"]
    assert "Q:门锁保修多久" in out["memory_context"]
    assert "保修 2 年" in out["memory_context"]


# ---- memory_agent.record：把本轮回答追加为 assistant 消息（供 checkpointer 累积）----
@pytest.mark.asyncio
async def test_memory_agent_record_appends_assistant(tmp_path):
    services = _FakeServices(str(tmp_path))
    out = await memory_agent.record(
        services, {"question": "门锁想换货", "final_answer": "可以换货，请提供订单号"}
    )
    assert out["memory_written"] is True
    assert out["messages"] == [{"role": "assistant", "content": "可以换货，请提供订单号"}]


# ---- reducer 跨轮累积：Annotated[operator.add] 让 checkpointer 累积历史对话 ----
@pytest.mark.asyncio
async def test_messages_reducer_accumulates_across_turns():
    graph = StateGraph(AgentState)
    graph.add_node("record", lambda s: {"messages": [{"role": "assistant", "content": "答"}]})
    graph.add_edge(START, "record")
    graph.add_edge("record", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    await compiled.ainvoke({"messages": [{"role": "user", "content": "问1"}]}, config)
    r2 = await compiled.ainvoke({"messages": [{"role": "user", "content": "问2"}]}, config)
    roles = [m["role"] for m in r2["messages"]]
    # 第2轮 = 第1轮的 [user, assistant] + 本轮的 [user, assistant]
    assert roles == ["user", "assistant", "user", "assistant"]
