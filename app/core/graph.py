"""LangGraph 主编排 —— 主控路由 + 子 Agent 节点 + 共享状态（= 团队记忆）+ checkpointer。

这是把「多智能体 + 共享记忆」串起来的关键。三个成对的绑定（面试重点讲）：

1. **共享 State = 团队/协调记忆**：所有子 Agent 节点读写同一个 `AgentState`，
   LangGraph 用 checkpointer 按 thread_id 持久化它（跨会话不丢）。
   这直接对应用户痛点④「多智能体记忆孤岛」—— 异构 Agent（qa/memory/tools/reviewer）
   共享同一份状态，不会各说各话。它们与「用户长期记忆」（episodic/semantic）是两层：
     - 团队记忆（共享 State）：本次任务/会话内，各 Agent 的中间结果与决策。
     - 用户长期记忆（MemoryStore）：跨会话沉淀的用户事实/偏好。
2. **子 Agent 模块化**：每个节点是 `subagents/` 里一个纯函数，状态读写与流程解耦，
   加一个新子 Agent = 写一个节点 + 连一条边（面试讲"为什么拆子 Agent"）。
3. **优雅降级**：任何节点/组件失败都降级（TF-IDF、无评审、无模型→片段），保证任意环境能跑。

流程：START → memory(检索长期记忆) → route(意图路由) → qa/tools/escalate → safety(安检)
        → review(模型B评审) → record(写情景记忆) → END
"""
from __future__ import annotations

import logging
import operator
import re
from functools import partial
from typing import Annotated, Any, AsyncIterator, Dict, List, Literal, TypedDict

from ..config import settings
from ..subagents import memory_agent, qa, reviewer, tools
from .services import Services, get_services

logger = logging.getLogger("agent.graph")


# ============================================================
#  共享状态（团队/协调记忆）—— 所有子 Agent 读写同一份
# ============================================================
class AgentState(TypedDict, total=False):
    question: str
    request_id: str
    thread_id: str                  # 用户/会话标识，用于长期记忆的跨用户隔离（P0 修复）
    route: str                      # "qa" | "tools"
    memory_context: str             # 用户长期记忆（记忆子 Agent 注入）
    memory_hit: bool
    retrieved_docs: list[Dict[str, Any]]
    answer: str                     # 模型A 原始回答
    final_answer: str
    review: Dict[str, Any]
    review_comment: str
    need_revision: bool
    safety: Dict[str, Any]
    sources: list[str]
    messages: Annotated[list[Dict[str, Any]], operator.add]  # 工作记忆（checkpointer 跨轮累积）


# ============================================================
#  意图路由（售后场景：查订单 / 查物流 / 退款 → tools；否则 qa）
# ============================================================
def _route(state: AgentState) -> str:
    question = state.get("question") or ""
    # 投诉诉求优先级最高：先转人工，再判工具，最后 QA（Day 13 补「投诉→转人工」路由）
    if tools.resolve_escalation(question):
        return "escalate"
    tool_name, _ = tools.resolve_tool_call(question)
    return "tools" if tool_name else "qa"


def route_node(state: AgentState) -> Dict[str, Any]:
    return {"route": _route(state)}


# ============================================================
#  子 Agent 节点（薄封装，逻辑在 subagents/*）
# ============================================================
async def memory_node(services: Services, state: AgentState) -> Dict[str, Any]:
    return await memory_agent.run(services, state)


async def qa_node(services: Services, state: AgentState) -> Dict[str, Any]:
    return await qa.run(services, state)


async def escalate_node(services: Services, state: AgentState) -> Dict[str, Any]:
    """投诉转人工：返回固定话术（无需 LLM，转交人工客服）。"""
    return {"answer": tools.format_escalation()}


async def tools_node(services: Services, state: AgentState) -> Dict[str, Any]:
    question = state.get("question", "")
    # 加分项：开 TOOL_USE_FUNCTION_CALLING 时优先走 bind_tools Function Calling，
    # 模型自主决定调哪个工具（tool_call→执行→回填→续生成）；失败/无模型自动退回启发式路由。
    if settings.tool_use_function_calling:
        fc_answer = await tools.run_with_function_calling(services, question)
        if fc_answer is not None:
            return {"answer": fc_answer}
    # 默认：启发式路由（查订单/查物流/退款 → 对应售后 mock 工具，走白名单调度）
    tool_name, args = tools.resolve_tool_call(question)
    if tool_name is None:
        return {"answer": "未识别到可执行工具，请提供订单号 / 物流单号 / 退款诉求。"}
    tool_result = await tools.run_tool(services, tool_name, args)
    return {"answer": tools.format_tool_result(tool_result)}


async def safety_node(services: Services, state: AgentState) -> Dict[str, Any]:
    text = state.get("answer", "") or state.get("final_answer", "")
    result = await tools.safety_gate(services, text)
    return {"safety": result}


async def review_node(services: Services, state: AgentState) -> Dict[str, Any]:
    return await reviewer.run(services, state)


async def record_node(services: Services, state: AgentState) -> Dict[str, Any]:
    return await memory_agent.record(services, state)


# ============================================================
#  图构建
# ============================================================
# 全局持有 sqlite 连接（aiosqlite.Connection），供 shutdown 时关闭
_checkpointer_conn: Any = None


async def _build_checkpointer():
    """优先 SQLite（重启不丢），失败降级内存。

    ⚠️ 坑（B 的修复点，A 实测抓到的真根因）：
      1. LangGraph 的 checkpointer 分两套：`SqliteSaver`（同步）与 `AsyncSqliteSaver`（异步）。
         我们用 `graph.ainvoke`/`astream`（异步），必须配 **AsyncSqliteSaver**，
         否则会报 "SqliteSaver does not support async methods"。
      2. `AsyncSqliteSaver.from_conn_string(path)` 返回的是**上下文管理器**，需 `async with` 进入，
         不能直接当 checkpointer 传。所以改为手动 `aiosqlite.connect` 再 `AsyncSqliteSaver(conn)`，
         连接由我们持有、可显式关闭。
    """
    global _checkpointer_conn
    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        _checkpointer_conn = await aiosqlite.connect(settings.sqlite_memory_path)
        logger.info("AsyncSqliteSaver 已连接：%s（重启不丢，跨会话工作记忆持久化）", settings.sqlite_memory_path)
        return AsyncSqliteSaver(_checkpointer_conn)
    except Exception as e:
        logger.warning("AsyncSqliteSaver 不可用（%s），降级 InMemorySaver（重启不丢工作记忆）", e)
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            return InMemorySaver()
        except Exception:
            return None


async def close_checkpointer() -> None:
    """shutdown 时关闭 aiosqlite 连接。"""
    global _checkpointer_conn
    if _checkpointer_conn is not None:
        try:
            await _checkpointer_conn.close()
        except Exception:
            pass
        _checkpointer_conn = None


async def build_graph(start: bool = True):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as e:
        raise RuntimeError("需要安装 langgraph（pip install langgraph）") from e

    services = get_services()

    builder = StateGraph(AgentState)
    # 用 functools.partial 绑定 services：它保留 awaitable 属性，LangGraph 才能正确 await
    builder.add_node("memory", partial(memory_node, services))
    builder.add_node("route", route_node)
    builder.add_node("qa", partial(qa_node, services))
    builder.add_node("tools", partial(tools_node, services))
    builder.add_node("escalate", partial(escalate_node, services))
    builder.add_node("safety", partial(safety_node, services))
    builder.add_node("review", partial(review_node, services))
    builder.add_node("record", partial(record_node, services))

    # 边
    builder.add_edge(START, "memory")
    builder.add_edge("memory", "route")
    builder.add_conditional_edges(
        "route", lambda s: s.get("route", "qa"),
        {"qa": "qa", "tools": "tools", "escalate": "escalate"},
    )
    builder.add_edge("qa", "safety")
    builder.add_edge("tools", "safety")
    builder.add_edge("escalate", "safety")

    # 安检：不安全 → 提前结束；安全 → 分流
    def _after_safety(s: AgentState) -> str:
        if (s.get("safety") or {}).get("safe") is False and settings.safety_enabled:
            return "blocked"
        # 工具结果 / 转人工话术是「后端权威数据 / 固定话术」，无需模型 B 评审二次猜测
        # （评审只针对自由生成的 QA 答案，否则评审会把「订单已签收」这类确定性结果误判为编造改写）。
        if s.get("route") in ("tools", "escalate"):
            return "record"
        return "review"

    builder.add_conditional_edges(
        "safety",
        _after_safety,
        {"blocked": END, "review": "review", "record": "record"},
    )
    builder.add_edge("review", "record")
    builder.add_edge("record", END)

    checkpointer = await _build_checkpointer()
    if checkpointer is not None:
        graph = builder.compile(checkpointer=checkpointer)
    else:
        graph = builder.compile()
    logger.info("LangGraph 已编译（节点: memory/route/qa/tools/safety/review/record）")
    return graph


# 全局单例（lifespan 时构建；import 失败给 None，由调用方降级）
_graph: Any = None
_graph_error: str | None = None


async def get_graph():
    global _graph, _graph_error
    if _graph is None and _graph_error is None:
        try:
            _graph = await build_graph()
        except Exception as e:
            _graph_error = str(e)
            logger.warning("LangGraph 构建失败（%s），调用方将降级为直连", e)
    return _graph


def graph_status() -> Dict[str, Any]:
    return {"compiled": _graph is not None, "error": _graph_error}


# ============================================================
#  执行入口（供路由层调用）
# ============================================================
async def run_agent(question: str, request_id: str = "", thread_id: str = "") -> Dict[str, Any]:
    """执行一次问答，返回最终 State（含答案/来源/评审/记忆命中）。"""
    graph = await get_graph()
    # 与 checkpointer 的 thread_id 对齐：作为长期记忆跨用户隔离的 key（P0 修复）
    tid = thread_id or request_id
    initial: AgentState = {
        "question": question,
        "request_id": request_id,
        "thread_id": tid,
        "messages": [{"role": "user", "content": question}],
    }
    config = {"configurable": {"thread_id": tid}}
    if graph is None:
        # 降级：无 graph（缺 langgraph）时，用直连子 Agent 跑最小流程 QA
        services = get_services()
        st = await qa_node(services, initial)
        return {**initial, **st, "review_comment": "LangGraph 不可用（直连降级）"}
    result = await graph.ainvoke(initial, config)
    return result


async def run_agent_stream(question: str, request_id: str = "", thread_id: str = "") -> AsyncIterator[dict[str, Any]]:
    """流式执行：产出事件字典序列（供 SSE）。"""
    graph = await get_graph()
    if graph is None:
        yield {"type": "token", "content": "LangGraph 未配置，无法流式。"}
        return
    tid = thread_id or request_id
    initial: AgentState = {
        "question": question,
        "request_id": request_id,
        "thread_id": tid,
        "messages": [{"role": "user", "content": question}],
    }
    config = {"configurable": {"thread_id": tid}}
    async for event in graph.astream(initial, config):
        # event 形如 {"node_name": {"更新后的部分字段"}}，简化成通用事件
        for node_name, updates in event.items():
            if isinstance(updates, dict):
                yield {"type": "node", "node": node_name, "updates": updates}
                # 评审节点完成时，额外产出结构化 review 事件（供前端/客户端单独消费）
                if node_name == "review":
                    yield {
                        "type": "review",
                        "review_score": (updates.get("review") or {}).get("overall"),
                        "review_comment": updates.get("review_comment", ""),
                        "need_revision": bool(updates.get("need_revision", False)),
                        "final_answer": updates.get("final_answer", ""),
                    }
