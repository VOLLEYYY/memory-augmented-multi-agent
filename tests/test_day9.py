"""Day 9 单测 —— 覆盖「跨用户记忆隔离（P0 修复）」+「售后工具 + 白名单 + 路由」。

零重依赖：不碰 qdrant / sentence-transformers / torch / 真实模型。
- 隔离逻辑用 `_InMemoryEpisodic`（episodic 的内存降级实现）测过滤行为；
- 工具用 mock 工具函数测调度、白名单拒绝、意图路由。
运行：cd multi-agent && pytest -q tests/test_day9.py
"""
from __future__ import annotations

import pytest


# ============================================================
#  一、跨用户记忆隔离（P0 修复）
# ============================================================
@pytest.mark.asyncio
async def test_episodic_inmemory_thread_isolation():
    """给定 thread_id 时，只召回该用户的记忆，不串其它用户。"""
    from app.memory.episodic import _InMemoryEpisodic
    from app.memory.models import MemoryItem, MemoryType

    store = _InMemoryEpisodic()
    await store.add(MemoryItem(
        id="u1", type=MemoryType.EPISODIC, content="智能门锁保修多久",
        metadata={"thread_id": "user-1"},
    ))
    await store.add(MemoryItem(
        id="u2", type=MemoryType.EPISODIC, content="智能门锁怎么换货",
        metadata={"thread_id": "user-2"},
    ))

    # user-2 检索「门锁」，只命中 user-2 的记忆（不串 user-1 的「保修」记忆）
    r2 = await store.search("门锁", k=5, thread_id="user-2")
    assert r2, "user-2 应命中自己的记忆"
    assert all(it.metadata.get("thread_id") == "user-2" for it in r2)
    assert any("换货" in it.content for it in r2)

    # 不传 thread_id 时仍全局召回（兼容旧调用，如固化 consolidate 的 list_recent）
    rall = await store.search("门锁", k=5)
    assert len(rall) >= 2


@pytest.mark.asyncio
async def test_memory_agent_run_passes_thread_id_to_search():
    """memory_agent.run 检索 episodic 时带上 thread_id 过滤。"""
    from app.subagents import memory_agent

    class FakeEpisodic:
        def __init__(self):
            self.captured_thread_id = None

        async def search(self, query, thread_id=None):
            self.captured_thread_id = thread_id
            from app.memory.models import MemoryRetrieval
            return MemoryRetrieval(items=[], hit=False)

    class FakeSemantic:
        async def search(self, query, key=None, k=5):
            from app.memory.models import MemoryRetrieval
            return MemoryRetrieval(items=[], hit=False)

    services = type("S", (), {"episodic": FakeEpisodic(), "semantic": FakeSemantic()})()
    await memory_agent.run(services, {"question": "门锁", "thread_id": "user-9", "messages": []})
    assert services.episodic.captured_thread_id == "user-9"


@pytest.mark.asyncio
async def test_memory_agent_record_writes_thread_id():
    """memory_agent.record 写 episodic 时把 thread_id 落进 metadata。"""
    from app.subagents import memory_agent

    class FakeEpisodic:
        def __init__(self):
            self.items = []

        async def add(self, item):
            self.items.append(item)

    services = type("S", (), {"episodic": FakeEpisodic()})()
    out = await memory_agent.record(
        services, {"question": "门锁", "answer": "整机保修1年", "thread_id": "user-9"}
    )
    assert out["memory_written"] is True
    assert services.episodic.items[0].metadata["thread_id"] == "user-9"


# ============================================================
#  二、售后工具：调度 + 白名单拒绝
# ============================================================
@pytest.mark.asyncio
async def test_run_tool_query_order():
    from app.subagents.tools import format_tool_result, run_tool

    result = await run_tool(None, "query_order", {"order_id": "A123"})
    assert result["tool"] == "query_order"
    assert result["status"] in ("待发货", "运输中", "已签收")
    # 同一订单号结果稳定（确定性 mock，而非随机状态）
    again = await run_tool(None, "query_order", {"order_id": "A123"})
    assert again["status"] == result["status"]
    assert "A123" in format_tool_result(result)


@pytest.mark.asyncio
async def test_run_tool_query_logistics():
    from app.subagents.tools import format_tool_result, run_tool

    result = await run_tool(None, "query_logistics", {"tracking_no": "SF1234567890"})
    assert result["tracking_no"] == "SF1234567890"
    assert result["traces"], "应包含物流轨迹"
    assert "SF1234567890" in format_tool_result(result)


@pytest.mark.asyncio
async def test_run_tool_apply_refund_returns_refund_no():
    from app.subagents.tools import format_tool_result, run_tool

    result = await run_tool(None, "apply_refund", {"order_id": "A123", "reason": "不喜欢"})
    assert result["refund_no"].startswith("RF")
    assert "退款单号" in format_tool_result(result)


@pytest.mark.asyncio
async def test_run_tool_whitelist_rejects():
    from app.subagents.tools import run_tool

    # weather 已注册但不在白名单 → 拒绝
    r1 = await run_tool(None, "weather", {"city": "北京"})
    assert "error" in r1 and "白名单" in r1["error"]

    # 未注册的未知工具 → 拒绝
    r2 = await run_tool(None, "delete_all", {})
    assert "error" in r2 and "未注册" in r2["error"]


# ============================================================
#  三、意图路由（售后关键词 → 工具）
# ============================================================
def test_resolve_tool_call_order():
    from app.subagents.tools import resolve_tool_call

    name, args = resolve_tool_call("查一下我的订单 A123")
    assert name == "query_order"
    assert args["order_id"] == "A123"


def test_resolve_tool_call_refund():
    from app.subagents.tools import resolve_tool_call

    name, args = resolve_tool_call("申请退款")
    assert name == "apply_refund"
    assert args["order_id"]  # 未给单号时也有兜底 order_id

    # 「对订单 A123 申请退款」同时含「订单」和「退款」，应走退款
    name2, args2 = resolve_tool_call("对订单 A123 申请退款")
    assert name2 == "apply_refund"
    assert args2["order_id"] == "A123"


def test_resolve_tool_call_logistics():
    from app.subagents.tools import resolve_tool_call

    name, args = resolve_tool_call("查一下物流 SF1234567890")
    assert name == "query_logistics"
    assert args["tracking_no"] == "SF1234567890"


def test_resolve_tool_call_falls_back_to_none():
    from app.subagents.tools import resolve_tool_call

    # 知识问答（非工具意图）→ 走 qa
    name, args = resolve_tool_call("门锁保修多久")
    assert name is None
