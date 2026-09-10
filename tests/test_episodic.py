"""情景记忆单测 —— 覆盖 Day 6「记忆②」的 TTL 过期判定与 importance 加权排序（纯逻辑）。

零依赖：只测 EpisodicMemory 的 `_is_expired` 与 `_weighted_score` 两个纯函数，
不碰 qdrant / sentence-transformers / torch（与 test_smoke.py 同一哲学）。
真实的 Qdrant 稠密写入 / 检索由 scripts/verify_episodic.py 验证。

运行：cd multi-agent && pytest -q tests/test_episodic.py
"""
from __future__ import annotations

import time


# ---- TTL：超过 memory_ttl_days 天判定过期 ----
def test_episodic_is_expired_boundary():
    from app.config import settings
    from app.memory.episodic import EpisodicMemory
    from app.memory.models import MemoryItem, MemoryType

    ep = EpisodicMemory()  # 只建对象，不加载 embedding / qdrant
    ttl = settings.memory_ttl_days * 86400
    now = time.time()

    fresh = MemoryItem(id="a", type=MemoryType.EPISODIC, content="x", timestamp=now)
    assert ep._is_expired(fresh) is False            # 刚写入不过期

    # 恰好在边界内（差 1 秒）不算过期；超过 TTL 才算
    edge = MemoryItem(id="b", type=MemoryType.EPISODIC, content="x",
                      timestamp=now - ttl + 1)
    assert ep._is_expired(edge) is False

    old = MemoryItem(id="c", type=MemoryType.EPISODIC, content="x",
                     timestamp=now - ttl - 1)
    assert ep._is_expired(old) is True               # 超 TTL 过期


# ---- importance 加权：同一相似度下，importance 高者得分更高 ----
def test_episodic_weighted_score_orders_by_importance():
    from app.memory.episodic import EpisodicMemory

    ep = EpisodicMemory()
    # 相似度相同，importance 高者得分高
    assert ep._weighted_score(0.8, 0.9) > ep._weighted_score(0.8, 0.1)
    # importance=0 时退化为纯相似度
    assert ep._weighted_score(0.8, 0.0) == 0.8
    # 相似度更高但 importance 略低，仍可能被高 importance 反超（加权非纯相似度）
    assert ep._weighted_score(0.7, 0.9) > ep._weighted_score(0.8, 0.0)


# ---- importance 越界被夹到 [0,1]，排序稳定不因脏数据崩 ----
def test_episodic_weighted_score_clamps_importance():
    from app.memory.episodic import EpisodicMemory

    ep = EpisodicMemory()
    # 越界 importance（1.5 / -0.5）应被夹到 1.0 / 0.0，与合法值等分
    assert ep._weighted_score(0.5, 1.5) == ep._weighted_score(0.5, 1.0)
    assert ep._weighted_score(0.5, -0.5) == ep._weighted_score(0.5, 0.0)
