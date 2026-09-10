"""情景记忆（长期，跨会话）—— 对话片段向量化存 Qdrant + 相似度检索 + TTL 淘汰。

设计（对齐 HELLO QdrantRetriever，但对象是 MemoryItem 而非知识块）：
1. 优先用 sentence-transformers 嵌入，存 Qdrant，HNSW 近似检索。
2. 任何环节失败（未装依赖 / 服务不可用 / 嵌入模型加载失败）自动降级为
   内存关键词检索（InMemoryEpisodic），保证任何环境能跑 —— 延续 HELLO 降级哲学。
3. 每条记忆带 `importance` 与 `timestamp`，TTL 淘汰 + 重要性加权检索排序。

返回统一为 :class:`MemoryRetrieval`，与 graph 解耦。
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import settings
from .models import MemoryItem, MemoryRetrieval, MemoryType

logger = logging.getLogger("agent.memory.episodic")

# importance 加权检索：先多取候选，再按「相似度 × (1 + w·importance)」重排，
# 让高重要记忆有机会超越「略相似但更不重要」的记忆（importance 是排序权重）。
_CANDIDATE_MULTIPLIER = 4


# ============================================================
#  嵌入器（自包含）：优先 ST，失败降级为简单哈希向量
# ============================================================
class _Embedder:
    """把文本映射为稠密向量。无 embedding 时退化为可用的稀疏/哈希向量，保证都能跑。"""

    def __init__(self, model_name: str = "", device: str = "cpu"):
        self._st = None
        self._st_model = model_name
        self._device = device
        self._can_st = False

    def lazy_load(self) -> bool:
        """延迟加载 sentence-transformers；成功则启用稠密向量。"""
        if self._can_st:
            return True
        if not self._st_model:
            return False
        try:
            from sentence_transformers import SentenceTransformer  # 延迟导入
            self._st = SentenceTransformer(self._st_model, device=self._device)
            self._can_st = True
            return True
        except Exception as e:  # 未安装 / 模型加载失败
            logger.warning("embedding 不可用（%s），情景记忆降级为关键词检索", e)
            self._can_st = False
            return False

    async def embed(self, text: str) -> List[float]:
        if self.lazy_load():
            vec = await asyncio.to_thread(self._st.encode, text, normalize_embeddings=True)
            return [float(v) for v in vec]
        # 降级：字符级哈希袋向量（固定 256 维，便于一致性）
        vec = [0.0] * 256
        for ch in set(text):
            vec[hash(ch) % 256] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


# ============================================================
#  降级实现：内存关键词检索（任何环境都能跑）
# ============================================================
class _InMemoryEpisodic:
    mode = "sparse"

    def __init__(self):
        self._items: List[MemoryItem] = []
        self._index: Dict[str, Dict[str, Any]] = {}  # id -> {vec}

    def _term_set(self, text: str) -> set:
        return set(text.lower())

    async def add(self, item: MemoryItem) -> None:
        self._items.append(item)
        self._index[item.id] = {"terms": self._term_set(item.content)}

    async def search(self, query: str, k: int, thread_id: Optional[str] = None) -> List[MemoryItem]:
        qterms = self._term_set(query)
        scored: List[tuple] = []
        for item in self._items:
            if self._is_expired(item):
                continue
            # 跨用户隔离：给定 thread_id 时只召回该用户的记忆
            if thread_id and item.metadata.get("thread_id") != thread_id:
                continue
            overlap = len(qterms & self._index[item.id]["terms"])
            if overlap > 0:
                scored.append((overlap + item.importance, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:k]]

    def _is_expired(self, item: MemoryItem) -> bool:
        ttl = settings.memory_ttl_days * 86400
        return (time.time() - item.timestamp) > ttl

    @property
    def _count(self) -> int:
        return len(self._items)


# ============================================================
#  主实现：Qdrant 稠密情景记忆
# ============================================================
class EpisodicMemory:
    """跨会话情景记忆：写入对话片段 -> 向量化 -> 按相似度检索。"""

    mode = "dense"

    def __init__(self, embedder: Optional[_Embedder] = None, collection: Optional[str] = None):
        self._embedder = embedder or _Embedder(settings.embedding_model, settings.embedding_device)
        self._client = None
        self._collection = collection or settings.memory_episodic_collection
        self._stores: Dict[str, Any] = {}  # 降级兜底：其它来源记忆

    # ---- 惰性 Qdrant：复用全局共享客户端（避免本地模式并发冲突）----
    @property
    def _qdrant(self):
        from ..core.qdrant import get_qdrant_client
        return get_qdrant_client()

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams
        if not self._qdrant.collection_exists(self._collection):
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def _is_expired(self, item: MemoryItem) -> bool:
        ttl = settings.memory_ttl_days * 86400
        return (time.time() - item.timestamp) > ttl

    @staticmethod
    def _weighted_score(similarity: float, importance: float) -> float:
        """importance 作为检索排序的加权项：相似度 × (1 + w·importance)。

        importance 被夹到 [0,1]（越界不影响排序稳定性）；w=0 时退化为纯相似度。
        """
        imp = max(0.0, min(1.0, importance))
        return similarity * (1.0 + settings.memory_importance_weight * imp)

    async def add(self, item: MemoryItem) -> None:
        """写入一条情景记忆。embedding 不可用时降级。"""
        try:
            if not self._embedder.lazy_load():
                store = self._stores.setdefault("_memory", _InMemoryEpisodic())
                await store.add(item)
                return
            from qdrant_client.models import PointStruct
            vec = await self._embedder.embed(item.content)
            self._ensure_collection(len(vec))
            # payload 用 content/source/type/importance/timestamp；id 用 item.id 的哈希
            self._qdrant.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=abs(hash(item.id)) % (2**63),
                        vector=vec,
                        payload=item.to_dict(),
                    )
                ],
            )
        except Exception as e:
            logger.warning("情景记忆写入失败（%s），降级到内存", e)
            store = self._stores.setdefault("_memory", _InMemoryEpisodic())
            await store.add(item)

    async def search(self, query: str, k: int | None = None,
                     thread_id: Optional[str] = None) -> MemoryRetrieval:
        """按语义相似度检索情景记忆。

        Args:
            thread_id: 给定则只在该用户（thread）的记忆范围内召回 —— 跨用户记忆隔离。
                这是 Day8 真实验证暴露的 P0 缺陷的修复：长期记忆（episodic）此前按全局
                语义检索，导致 user-2 会命中 user-1 的记忆（隐私泄露 + 回答串线）。
        """
        k = k or settings.memory_top_k
        try:
            if not self._embedder.lazy_load():
                store = self._stores.get("_memory")
                if not store:
                    return MemoryRetrieval(items=[], hit=False)
                items = await store.search(query, k, thread_id=thread_id)
                return MemoryRetrieval(items=items, hit=bool(items))
            qvec = await self._embedder.embed(query)
            if not self._qdrant.collection_exists(self._collection):
                return MemoryRetrieval(items=[], hit=False)
            # 跨用户隔离：payload 里 metadata.thread_id 精确匹配当前用户
            query_filter = None
            if thread_id:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                query_filter = Filter(
                    must=[FieldCondition(
                        key="metadata.thread_id", match=MatchValue(value=thread_id),
                    )]
                )
            hits = self._qdrant.query_points(
                collection_name=self._collection,
                query=qvec,
                query_filter=query_filter,
                limit=max(k * _CANDIDATE_MULTIPLIER, 10),  # 多取候选，供 importance 重排
                with_payload=True,
            ).points
            # 过期过滤 + importance 加权重排（相似度 × (1 + w·importance)）
            weighted: List[tuple] = []
            for h in hits:
                item = MemoryItem.from_dict(h.payload)
                if h.payload.get("timestamp") and self._is_expired(item):
                    continue
                sim = float(getattr(h, "score", 0.0) or 0.0)
                weighted.append((self._weighted_score(sim, item.importance), item))
            weighted.sort(key=lambda x: x[0], reverse=True)
            items = [it for _, it in weighted[:k]]
            return MemoryRetrieval(items=items, hit=bool(items))
        except Exception as e:
            logger.warning("情景记忆检索失败（%s），降级到内存", e)
            store = self._stores.get("_memory")
            if not store:
                return MemoryRetrieval(items=[], hit=False)
            items = await store.search(query, k, thread_id=thread_id)
            return MemoryRetrieval(items=items, hit=bool(items))

    async def list_recent(self, n: int = 20) -> List[MemoryItem]:
        """按时间戳降序拉取最近 n 条情景记忆（供固化子 Agent 使用）。

        与 search() 的区别：search 按「语义相似度」检索，适合回答时找相关内容；
        固化（consolidate）需要的是「近期发生了什么」，应按时间序拉取，而非相似度。
        """
        try:
            if not self._embedder.lazy_load():
                store = self._stores.get("_memory")
                if not store:
                    return []
                items = sorted(store._items, key=lambda it: it.timestamp, reverse=True)
                return items[:n]
            if not self._qdrant.collection_exists(self._collection):
                return []
            points, _ = self._qdrant.scroll(
                collection_name=self._collection,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            items = [MemoryItem.from_dict(p.payload) for p in points]
            # 过滤过期 + 按 timestamp 降序
            items = [it for it in items if not self._is_expired(it)]
            items.sort(key=lambda it: it.timestamp, reverse=True)
            return items[:n]
        except Exception as e:
            logger.warning("拉取近期情景记忆失败（%s）", e)
            return []


# 全局单例（惰性），避免每次请求重建
_instance: Optional[EpisodicMemory] = None


def get_episodic_memory() -> EpisodicMemory:
    global _instance
    if _instance is None:
        _instance = EpisodicMemory()
    return _instance
