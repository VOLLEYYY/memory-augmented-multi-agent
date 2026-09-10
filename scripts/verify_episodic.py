"""Day6 验证脚本：episodic 真向量化 + TTL 过期过滤 + importance 加权排序。

跑法（在项目根目录下，务必设 PYTHONPATH，否则 ModuleNotFoundError: app）：
    PYTHONPATH=. python scripts/verify_episodic.py

验证四项：
  1. embedder.lazy_load() == True（走 Qdrant 稠密，不是内存/哈希降级）
  2. 真集合 agent_memory_episodic 里已有真实向量（历史 ask 写入）
  3. TTL：把某条记忆 timestamp 改老（>30 天），检索时被 _is_expired 过滤
  4. importance：同 query 写两条内容相同、importance 不同的记忆，高者稳定排前

为不污染真集合，TTL / importance 的「写-查」都在独立集合 <collection>_verify 上做，跑完删除。
"""
from __future__ import annotations

import asyncio
import sys
import time

# Windows 控制台默认 GBK，打 emoji/中文会 UnicodeEncodeError；统一强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import settings
from app.memory.episodic import EpisodicMemory, get_episodic_memory
from app.memory.models import MemoryItem, MemoryType


async def main() -> None:
    print("=" * 64)
    print(f"embedding_model        = {settings.embedding_model}")
    print(f"memory_episodic_collection = {settings.memory_episodic_collection}")
    print(f"memory_ttl_days        = {settings.memory_ttl_days}")
    print(f"memory_importance_weight  = {settings.memory_importance_weight}")
    print("=" * 64)

    ep = get_episodic_memory()

    # ---- 1) embedder 真加载（决定 dense / 内存降级）----
    t0 = time.time()
    ok = ep._embedder.lazy_load()
    print(f"[1] embedder.lazy_load() -> {ok}  (耗时 {time.time()-t0:.1f}s)")
    print(f"    episodic.mode = {ep.mode}")
    if not ok:
        print("!! 稠密不可用，episodic 会降级内存（跨进程会丢），终止验证")
        return

    # ---- 2) 真集合里已有哪些真实向量记忆 ----
    from app.core.qdrant import get_qdrant_client
    client = get_qdrant_client()
    real = settings.memory_episodic_collection
    if client.collection_exists(real):
        cnt = client.count(collection_name=real, exact=True).count
        print(f"[2] 真集合 {real} 已有 {cnt} 条真实向量记忆（历史 ask 写入）")
    else:
        print(f"[2] 真集合 {real} 尚不存在（尚未 ask 过）")

    # ---- 3) TTL + 4) importance：在独立验证集合上做，跑完删除 ----
    # 集合名带时间戳，避免本地模式 delete_collection 的 flush 时序 + 跨进程 hash 随机化
    # 导致「上次残留 + 本次写入」重复污染（可重复跑）。
    verify_col = f"{real}_verify_{int(time.time())}"
    epv = EpisodicMemory(collection=verify_col)

    now = time.time()
    ttl_sec = settings.memory_ttl_days * 86400
    items = [
        # TTL：内容一样，一条正常、一条 31 天前（过期）
        MemoryItem(id="v_fresh", type=MemoryType.EPISODIC,
                   content="用户想退货：智能门锁 A100 保修期 2 年",
                   importance=0.5, timestamp=now),
        MemoryItem(id="v_expired", type=MemoryType.EPISODIC,
                   content="用户想退货：智能门锁 A100 保修期 2 年",
                   importance=0.9, timestamp=now - ttl_sec - 100),
        # importance：内容完全相同，仅 importance 不同
        MemoryItem(id="v_imp_hi", type=MemoryType.EPISODIC,
                   content="用户想换货：智能门锁 A100",
                   importance=0.95, timestamp=now),
        MemoryItem(id="v_imp_lo", type=MemoryType.EPISODIC,
                   content="用户想换货：智能门锁 A100",
                   importance=0.1, timestamp=now),
    ]
    for it in items:
        await epv.add(it)
    print(f"[3] 已向验证集合 {verify_col} 写入 4 条记忆（含 1 条过期、2 条同内容不同 importance）")

    # ---- TTL 验证 ----
    res = await epv.search("智能门锁 A100 退货 保修", k=10)
    ids = [it.id for it in res.items]
    print(f"[4] 退货检索命中 ids = {ids}")
    assert "v_expired" not in ids, f"TTL 未生效：过期条目 v_expired 仍被检索到（{ids}）"
    assert "v_fresh" in ids, f"正常条目 v_fresh 应被检索到（{ids}）"
    print("    ✅ TTL 过期条目 v_expired 已被 _is_expired 过滤")

    # ---- importance 验证 ----
    res2 = await epv.search("智能门锁 A100 换货", k=10)
    ids2 = [it.id for it in res2.items]
    print(f"[5] 换货检索顺序 = {ids2}")
    assert "v_imp_hi" in ids2 and "v_imp_lo" in ids2, f"两条 importance 记忆应同时命中（{ids2}）"
    assert ids2.index("v_imp_hi") < ids2.index("v_imp_lo"), \
        f"importance 未影响排序：hi 应排在 lo 前（{ids2}）"
    print("    ✅ importance 高者（0.95）稳定排在低者（0.1）前")

    # ---- 清理 ----
    client.delete_collection(verify_col)
    print(f"[6] 已清理验证集合 {verify_col}")
    print("\n全部通过 ✅：episodic 真向量存 Qdrant / TTL 过滤 / importance 排序")


if __name__ == "__main__":
    asyncio.run(main())
