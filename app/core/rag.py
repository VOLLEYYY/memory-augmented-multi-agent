"""RAG 检索器 — 混合检索（BM25 + 稠密向量 → RRF 融合 → 可选重排）。

对齐 HELLO 的接口哲学：统一 `async search(query, top_k) -> [{content, score, source}]`，
上层只依赖这一个接口，可无缝切换/降级。

三层：
1. DenseRetriever —— 稠密检索（Embedder + Qdrant，复用 HELLO 思路）。
2. BM25Retriever —— 稀疏检索（自实现 BM25，零外部依赖，任何环境能跑）。
3. HybridRetriever —— 二者用 RRF（Reciprocal Rank Fusion）融合；可选 cross-encoder 重排。

降级链路（与 HELLO 一致）：Hybrid 任一环节失败 → 降级 BM25 → 再失败 TFIDF。
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import settings

logger = logging.getLogger("agent.rag")


# ============================================================
#  共享工具：分词 + 分块（复用 HELLO）
# ============================================================
def _tokenize(text: str) -> List[str]:
    """切 token：英文/数字按词，中文按单字 + 相邻 bigram。"""
    text = text.lower()
    tokens: List[str] = []
    tokens.extend(m.group() for m in re.finditer(r"[a-z0-9]+", text))
    cjk = re.findall(r"[一-鿿]", text)
    tokens.extend(cjk)
    tokens.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return tokens


def _split(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


# ============================================================
#  Embedder（自包含：优先 ST，降级哈希袋）
# ============================================================
class Embedder:
    def __init__(self, model_name: str = "", device: str = "cpu"):
        self._st = None
        self._model_name = model_name
        self._device = device
        self._can_st = False

    def lazy_load(self) -> bool:
        if self._can_st:
            return True
        if not self._model_name:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self._model_name, device=self._device)
            self._can_st = True
            return True
        except Exception as e:
            logger.warning("embedding 不可用（%s），稠密检索降级", e)
            self._can_st = False
            return False

    async def embed_query(self, text: str) -> List[float]:
        if self.lazy_load():
            vec = await asyncio.to_thread(self._st.encode, text, normalize_embeddings=True)
            return [float(v) for v in vec]
        return self._fallback_vec(text)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.lazy_load():
            vecs = await asyncio.to_thread(self._st.encode, texts, normalize_embeddings=True)
            return [[float(v) for v in vec] for vec in vecs]
        return [self._fallback_vec(t) for t in texts]

    @staticmethod
    def _fallback_vec(text: str) -> List[float]:
        vec = [0.0] * 256
        for ch in set(text.lower()):
            vec[hash(ch) % 256] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ============================================================
#  BM25Retriever：稀疏检索（首备）
# ============================================================
class BM25Retriever:
    mode = "bm25"

    def __init__(self, knowledge_base_path: str = "", chunk_size: int = 512,
                 chunk_overlap: int = 50):
        self.kb_path = Path(knowledge_base_path or settings.knowledge_base_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: List[Dict[str, Any]] = []
        self._doc_len: List[int] = []
        self._df: Dict[str, int] = {}
        self._avgdl = 0.0

    def ingest_directory(self, dir_path: str) -> int:
        path = Path(dir_path if dir_path else self.kb_path)
        if not path.is_dir():
            return 0
        self._chunks.clear()
        self._doc_len.clear()
        self._df.clear()
        for f in sorted(path.glob("*")):
            if f.suffix.lower() not in (".md", ".txt"):
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            for chunk in _split(text, self.chunk_size, self.chunk_overlap):
                token_set = set(_tokenize(chunk))
                self._chunks.append({
                    "content": chunk, "source": f.name,
                    "tokens": _tokenize(chunk),
                })
                self._doc_len.append(len(token_set))
                for t in token_set:
                    self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0
        return len(self._chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _bm25(self, tokens: List[str], dtf, doc_len: int) -> float:
        k1, b = 1.5, 0.75
        score = 0.0
        for t in tokens:
            tf = dtf.get(t, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (len(self._chunks) - self._df.get(t, 0) + 0.5)
                           / (self._df.get(t, 0) + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / (self._avgdl or 1))
            score += idf * (tf * (k1 + 1)) / denom
        return score

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._chunks:
            return []
        q = _tokenize(query)
        scored: List[tuple] = []
        for i, c in enumerate(self._chunks):
            dtf = {}
            for t in c["tokens"]:
                dtf[t] = dtf.get(t, 0) + 1
            s = self._bm25(q, dtf, self._doc_len[i])
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"content": c["content"], "score": round(s, 4), "source": c["source"]}
            for s, c in scored[:top_k]
        ]


# ============================================================
#  DenseRetriever：稠密检索（Qdrant + embedding）
# ============================================================
class DenseRetriever:
    mode = "dense"

    def __init__(self, embedder: Optional[Embedder] = None,
                 collection: str = "", url: str = "", local_path: str = "",
                 chunk_size: int = 512, chunk_overlap: int = 50):
        self.embedder = embedder or Embedder(settings.embedding_model)
        self.collection = collection or settings.qdrant_collection
        self.url = url or settings.qdrant_url
        self.local_path = local_path or settings.qdrant_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._client = None

    @property
    def _qdrant(self):
        # 用全局共享客户端：避免两个本地客户端撞同一个文件夹（"already accessed" 错误）
        from .qdrant import get_qdrant_client
        return get_qdrant_client()

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams
        if not self._qdrant.collection_exists(self.collection):
            self._qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def ingest_directory(self, dir_path: str, force: bool = False) -> int:
        from qdrant_client.models import PointStruct
        client = self._qdrant
        if force and client.collection_exists(self.collection):
            client.delete_collection(self.collection)
        chunks = []
        path = Path(dir_path)
        if path.is_dir():
            for f in sorted(path.glob("*")):
                if f.suffix.lower() not in (".md", ".txt"):
                    continue
                text = f.read_text(encoding="utf-8", errors="ignore")
                for chunk in _split(text, self.chunk_size, self.chunk_overlap):
                    if chunk.strip():
                        chunks.append({"content": chunk, "source": f.name})
        if not chunks:
            return 0
        vectors = await self.embedder.embed_texts([c["content"] for c in chunks])
        self._ensure_collection(len(vectors[0]))
        points = [
            PointStruct(id=i, vector=v, payload={"content": c["content"], "source": c["source"]})
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        client.upsert(collection_name=self.collection, points=points)
        return len(points)

    @property
    def chunk_count(self) -> int:
        try:
            return self._qdrant.count(collection_name=self.collection, exact=True).count
        except Exception:
            return 0

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        client = self._qdrant
        if not client.collection_exists(self.collection):
            return []
        qvec = await self.embedder.embed_query(query)
        hits = client.query_points(
            collection_name=self.collection, query=qvec,
            limit=top_k, with_payload=True,
        ).points
        return [
            {"content": h.payload["content"], "score": round(h.score, 4),
             "source": h.payload.get("source")}
            for h in hits
        ]


# ============================================================
#  HybridRetriever：BM25 + 稠密 → RRF 融合 → 可选重排
# ============================================================
class HybridRetriever:
    mode = "hybrid"

    def __init__(self, dense: Optional[DenseRetriever] = None,
                 bm25: Optional[BM25Retriever] = None,
                 top_k: int = 5, c=60, cross_encoder: str = ""):
        self.dense = dense
        self.bm25 = bm25 or BM25Retriever()
        self.top_k = top_k
        self.c = c  # RRF 常数
        self._cross_encoder_model = cross_encoder
        self._ce = None

    def _rrf_merge(self, lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """RRF（Reciprocal Rank Fusion）：按名次（而非分数）融合不同检索结果。

        ⚠️ doc 是 dict，不可哈希，不能直接当 dict key（否则 unhashable type: 'dict'）。
        用 (source, content) 作 key 去重合并。
        """
        scores: Dict[tuple, float] = {}
        best_doc: Dict[tuple, Dict[str, Any]] = {}
        for doc_list in lists:
            for rank, doc in enumerate(doc_list):
                key = (doc.get("source"), doc.get("content"))
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.c + rank + 1)
                best_doc.setdefault(key, doc)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [{**best_doc[key], "score": round(s, 4)} for key, s in ranked[: self.top_k]]

    def _maybe_rerank(self, docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """可选：cross-encoder 重排（未安装/未启用时原样返回）。"""
        if not self._cross_encoder_model or len(docs) <= 1:
            return docs
        try:
            if self._ce is None:
                from sentence_transformers import CrossEncoder, models  # noqa
                self._ce = CrossEncoder(self._cross_encoder_model)
            pairs = [(query, d["content"]) for d in docs]
            scores = self._ce.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [{**d, "score": round(float(s), 4)} for d, s in ranked]
        except Exception as e:
            logger.warning("cross-encoder 重排不可用（%s），跳过", e)
            return docs

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        candidates: List[List[Dict[str, Any]]] = []
        if self.dense is not None:
            try:
                candidates.append(await self.dense.search(query, top_k))
            except Exception as e:
                logger.warning("稠密检索失败（%s），用 BM25", e)
        try:
            candidates.append(await self.bm25.search(query, top_k))
        except Exception as e:
            logger.warning("BM25 失败（%s）：%s", e.__class__.__name__, e)
        if not candidates:
            return []
        fused = self._rrf_merge(candidates)
        return self._maybe_rerank(fused, query)


# ============================================================
#  工厂：混合优先，失败降级
# ============================================================
_built: Optional[Any] = None
_mode: str = "hybrid"


async def build_retriever(force_reload: bool = False):
    """构建检索器：Hybrid → BM25 降级。返回 (retriever, mode)。"""
    global _built, _mode
    if _built is not None and not force_reload:
        return _built, _mode
    try:
        embedder = Embedder(settings.embedding_model, settings.embedding_device)
        dense = DenseRetriever(
            embedder=embedder, collection=settings.qdrant_collection,
            url=settings.qdrant_url, local_path=settings.qdrant_path,
            chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap,
        )
        bm25 = BM25Retriever(chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap)
        n = bm25.ingest_directory(settings.knowledge_base_path)
        if settings.use_dense_retrieval:
            try:
                await dense.ingest_directory(settings.knowledge_base_path, force=force_reload)
            except Exception as e:
                logger.warning("稠密索引失败（%s），退化为 BM25 混合", e)
                dense = None
        _built = HybridRetriever(dense=dense, bm25=bm25, top_k=settings.top_k,
                                 cross_encoder=settings.rag_cross_encoder)
        _mode = "hybrid"
        logger.info("混合检索就绪，摄入 %s 个 chunk", n)
    except Exception as e:
        logger.warning("混合检索构建失败（%s），降级 BM25", e)
        _built = BM25Retriever(chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap)
        _built.ingest_directory(settings.knowledge_base_path)
        _mode = "bm25"
    return _built, _mode


def current_retriever_mode() -> str:
    return _mode


def get_retriever_sync() -> Any:
    return _built
