"""冒烟测试 —— 只测「零依赖可跑」的核心逻辑，不依赖 langgraph/qdrant/torch。

这些用例保证：即使模型/向量库不可用，项目最基础的逻辑依然正确（优雅降级的根基）。
运行：cd multi-agent && pytest -q （需先 pip install pytest pytest-asyncio pydantic-settings）
"""
from __future__ import annotations

import pytest

# ---- 安全 ----
def test_safety_tool_detects_pii_and_sensitive():
    import asyncio
    from app.tools.safety_tool import SafetyTool
    tool = SafetyTool()
    res = asyncio.run(tool.execute("正常内容"))
    assert res["safe"] is True
    res2 = asyncio.run(tool.execute("这是我的手机号 13812345678"))
    assert res2["safe"] is False
    assert res2["risk_level"] == "medium"


# ---- 记忆模型 ----
def test_memory_item_roundtrip():
    from app.memory.models import MemoryItem, MemoryType
    item = MemoryItem(id="x", type=MemoryType.EPISODIC, content="用户喜欢喝美式",
                      importance=0.8, source="qa", metadata={"tag": "pref"})
    d = item.to_dict()
    back = MemoryItem.from_dict(d)
    assert back.id == "x"
    assert back.type == MemoryType.EPISODIC
    assert back.importance == 0.8


# ---- 语义记忆（SQLite）----
@pytest.mark.asyncio
async def test_semantic_memory_upsert_search(tmp_path):
    from app.memory.semantic import SemanticMemory
    mem = SemanticMemory(path=str(tmp_path / "m.db"))
    await mem.upsert("user.hobby", "喜欢打篮球", source="user")
    await mem.upsert("user.hobby", "喜欢打羽毛球", source="user")  # 以新为准
    res = await mem.search("篮球")
    assert res.hit is True
    latest = [it for it in res.items if it.metadata.get("key") == "user.hobby"][0]
    assert "羽毛球" in latest.content  # 第二次覆盖
    mem.close()


# ---- 工作记忆截断 ----
def test_working_memory_trim():
    from app.memory.working import extract_recent_messages
    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": str(i)} for i in range(50)
    ]
    out = extract_recent_messages(msgs, keep=10)
    assert out[0] == msgs[0]  # 保头
    assert len(out) == 10


# ---- 分词 / 分块（RAG 纯函数）----
def test_tokenize_split():
    from app.core.rag import _tokenize, _split
    assert len(_tokenize("向量检索 deep learning 中文")) > 0
    chunks = _split("一二三四五六七八九十", 4, 1)
    assert len(chunks) >= 2


# ---- 记忆固化（降级：无 LLM 时规则拼接）----
@pytest.mark.asyncio
async def test_consolidator_rule_fallback(tmp_path, monkeypatch):
    from app.memory.models import MemoryItem, MemoryType
    from app.memory.consolidator import Consolidator
    from app.memory.episodic import _InMemoryEpisodic
    from app.memory.semantic import SemanticMemory

    sem = SemanticMemory(path=str(tmp_path / "c.db"))
    episodic = _InMemoryEpisodic()
    cons = Consolidator(episodic=episodic, semantic=sem)
    # 让 _get_llm 返回 None（模拟无模型）
    monkeypatch.setattr(cons, "_get_llm", lambda: None)

    items = [
        MemoryItem(id="1", type=MemoryType.EPISODIC, content="用户想学 LangGraph"),
        MemoryItem(id="2", type=MemoryType.EPISODIC, content="用户想学 LangGraph 编排"),
    ]
    result = await cons.consolidate(overrides=items)
    assert result.merged_count >= 0
    assert len(result.kept_memories) >= 1
