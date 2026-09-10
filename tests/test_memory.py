"""记忆模块单测 —— 覆盖 Day 5「记忆①」的模型与工作记忆截断逻辑。

零依赖、纯同步，不碰 langgraph/qdrant/torch（与 test_smoke.py 同一哲学）。
在 test_smoke.py 已有的 roundtrip / trim 基础上，补齐尚未覆盖的：
MemoryType 枚举、MemoryItem 默认值与序列化、MemoryRetrieval.as_content、
ConsolidationResult、recent_user_content、WorkingMemory.summarize。

运行：cd multi-agent && pytest -q tests/test_memory.py
"""
from __future__ import annotations


# ---- MemoryType 枚举：四种记忆各自用途（面试背这个）----
def test_memory_type_values():
    from app.memory.models import MemoryType
    assert MemoryType.WORKING.value == "working"        # 工作记忆：当前会话（checkpointer）
    assert MemoryType.EPISODIC.value == "episodic"      # 情景记忆：跨会话对话片段
    assert MemoryType.SEMANTIC.value == "semantic"      # 语义记忆：用户事实/偏好
    assert MemoryType.PROCEDURAL.value == "procedural"  # 程序记忆：学会的做法（后置）


# ---- MemoryItem：必填 + 默认值 ----
def test_memory_item_defaults():
    from app.memory.models import MemoryItem, MemoryType
    item = MemoryItem(id="m1", type=MemoryType.SEMANTIC, content="用户是金卡会员")
    assert item.importance == 0.5             # 默认权重
    assert item.source == "user"              # 默认来源
    assert item.metadata == {}                # 默认空扩展
    assert isinstance(item.timestamp, float)  # 默认时间戳（epoch 秒）


# ---- MemoryItem 序列化：type 存字符串；from_dict 缺字段回退默认 ----
def test_memory_item_to_from_dict():
    from app.memory.models import MemoryItem, MemoryType
    item = MemoryItem(id="m2", type=MemoryType.EPISODIC, content="退货工单 #88",
                      importance=0.9, source="tool", metadata={"order_id": "A123"})
    d = item.to_dict()
    assert d["type"] == "episodic"            # 枚举序列化为字符串
    assert d["metadata"] == {"order_id": "A123"}

    back = MemoryItem.from_dict({"id": "m3", "content": "x"})  # 只给必填
    assert back.type == MemoryType.EPISODIC   # 缺 type 默认 episodic
    assert back.importance == 0.5
    assert back.source == "user"
    assert back.metadata == {}


# ---- MemoryRetrieval：命中标记 + 注入文本格式 ----
def test_memory_retrieval_as_content():
    from app.memory.models import MemoryItem, MemoryType, MemoryRetrieval
    empty = MemoryRetrieval()
    assert empty.hit is False
    assert empty.as_content() == ""

    item = MemoryItem(id="r1", type=MemoryType.SEMANTIC, content="客户地址：北京",
                      importance=0.8, source="consolidator")
    retrieval = MemoryRetrieval(items=[item], hit=True)
    text = retrieval.as_content()
    assert "记忆·semantic" in text
    assert "重要0.8" in text
    assert "客户地址：北京" in text


# ---- ConsolidationResult：默认值 ----
def test_consolidation_result_defaults():
    from app.memory.models import ConsolidationResult
    r = ConsolidationResult()
    assert r.kept_memories == []
    assert r.merged_count == 0
    assert r.dropped_count == 0
    assert r.summary == ""


# ---- 工作记忆截断：空 / 不足 keep ----
def test_extract_recent_messages_empty_and_short():
    from app.memory.working import extract_recent_messages
    assert extract_recent_messages([]) == []
    msgs = [{"role": "system", "content": "s"}]
    assert extract_recent_messages(msgs, keep=20) == msgs  # 不足 keep 原样返回


# ---- 工作记忆截断：keep=2 是「保头 1 + 尾 1」的最小有效边界 ----
def test_extract_recent_messages_keep_two():
    from app.memory.working import extract_recent_messages
    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": str(i)} for i in range(10)
    ]
    out = extract_recent_messages(msgs, keep=2)
    assert out[0] == msgs[0]     # 保头（system 说明不丢）
    assert out[-1] == msgs[-1]   # 保尾（最近一条）
    assert len(out) == 2


# ---- recent_user_content：取最近一条 user 正文 ----
def test_recent_user_content():
    from app.memory.working import recent_user_content
    assert recent_user_content([]) == ""
    msgs = [
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "答"},
        {"role": "user", "content": "门锁怎么换货？"},
    ]
    assert recent_user_content(msgs) == "门锁怎么换货？"
    # 非字符串 content（多模态）能兜底转 str
    res = recent_user_content([{"role": "user", "content": ["文本", "图"]}])
    assert isinstance(res, str) and "文本" in res


# ---- WorkingMemory.summarize：抽 Q:A 对偶（注入上下文用）----
def test_working_memory_summarize():
    from app.memory.working import WorkingMemory
    wm = WorkingMemory()  # 默认 keep=20，消息少于此数不做截断
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    summary = wm.summarize(msgs)
    assert "Q:Q1" in summary and "A1" in summary
    assert "Q:Q2" in summary and "A2" in summary
    assert wm.current_question(msgs) == "Q2"  # 最近一条 user 问题
