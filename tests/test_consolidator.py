"""记忆固化子 Agent 单测 —— Day 7「记忆③」：摘要降级 + 分桶去重 + 矛盾合并。

零依赖：不碰 LLM/qdrant/torch。用 _InMemoryEpisodic 兜底 + monkeypatch `_get_llm` 返回 None，
只测 consolidate 的「规则降级」路径与 semantic 的「以新为准 + old_value」矛盾合并。
真实 LLM 摘要路径由 scripts 起服务验证（依赖 key）。

运行：cd multi-agent && pytest -q tests/test_consolidator.py
"""
from __future__ import annotations

import pytest

from app.memory.consolidator import Consolidator
from app.memory.episodic import _InMemoryEpisodic
from app.memory.models import MemoryItem, MemoryType
from app.memory.semantic import SemanticMemory


def _mk_cons(tmp_path, monkeypatch):
    sem = SemanticMemory(path=str(tmp_path / "c.db"))
    cons = Consolidator(episodic=_InMemoryEpisodic(), semantic=sem)
    monkeypatch.setattr(cons, "_get_llm", lambda: None)  # 模拟无模型 → 规则降级
    return cons, sem


# ---- _summarize_one：无 LLM 时规则降级（单条原样 / 多条标注合并）----
@pytest.mark.asyncio
async def test_summarize_one_rule_fallback(tmp_path, monkeypatch):
    cons, _ = _mk_cons(tmp_path, monkeypatch)
    single = [MemoryItem(id="1", type=MemoryType.EPISODIC, content="用户想学 LangGraph")]
    assert (await cons._summarize_one(single)) == "用户想学 LangGraph"
    multi = [
        MemoryItem(id="1", type=MemoryType.EPISODIC, content="用户想学 LangGraph"),
        MemoryItem(id="2", type=MemoryType.EPISODIC, content="用户想学 LangGraph 编排"),
    ]
    assert "合并自 2 条" in (await cons._summarize_one(multi))


# ---- _bucketize：相似内容合并到同一桶（去重）----
def test_bucketize_dedup(tmp_path, monkeypatch):
    cons, _ = _mk_cons(tmp_path, monkeypatch)
    items = [
        MemoryItem(id="1", type=MemoryType.EPISODIC, content="用户想学 LangGraph"),
        MemoryItem(id="2", type=MemoryType.EPISODIC, content="用户想学 LangGraph 编排"),
        MemoryItem(id="3", type=MemoryType.EPISODIC, content="用户想退货智能门锁 A100"),
    ]
    buckets = cons._bucketize(items)
    # 前两条字节相似度高 → 同桶；第三条不同 → 另开桶
    assert len(buckets) == 2


# ---- consolidate：无 LLM + overrides → 规则降级，产出 kept 摘要 + merged 计数 ----
@pytest.mark.asyncio
async def test_consolidate_rule_fallback(tmp_path, monkeypatch):
    cons, sem = _mk_cons(tmp_path, monkeypatch)
    items = [
        MemoryItem(id="1", type=MemoryType.EPISODIC, content="用户想学 LangGraph"),
        MemoryItem(id="2", type=MemoryType.EPISODIC, content="用户想学 LangGraph 编排"),
        MemoryItem(id="3", type=MemoryType.EPISODIC, content="用户想退货智能门锁 A100"),
    ]
    result = await cons.consolidate(overrides=items)
    assert result.merged_count >= 0
    assert len(result.kept_memories) >= 1
    assert "固化完成" in result.summary
    sem.close()


# ---- 矛盾合并：semantic 同 key 以新为准 + 保留 old_value（可审计）----
@pytest.mark.asyncio
async def test_semantic_conflict_merge_keeps_old_value(tmp_path):
    sem = SemanticMemory(path=str(tmp_path / "m.db"))
    await sem.upsert("user.address", "北京", source="user")
    await sem.upsert("user.address", "上海", source="user")  # 矛盾 → 以新为准
    res = await sem.search("address", key="user.address")
    item = res.items[0]
    assert "上海" in item.content                        # 以新为准
    assert item.metadata.get("old_value") == "北京"       # 旧值保留可审计
    assert item.metadata.get("update_count") == 2
    sem.close()
