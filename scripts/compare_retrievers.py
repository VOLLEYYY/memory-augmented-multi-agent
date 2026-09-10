"""Day4 三档检索对比：BM25-only vs hybrid(无重排) vs hybrid(cross-encoder 重排)。

产出：把每档每个 query 的 top-k 来源 + 分数写进 compare_result.json（UTF-8），
同时打印摘要。cross-encoder 首次会下载 bge-reranker-base（约 1.1GB）。
"""
import asyncio
import json
import time

from app.config import settings
from app.core.rag import BM25Retriever, DenseRetriever, Embedder, HybridRetriever

QUERIES = [
    "什么是混合检索？",
    "RAG 评估有哪些指标？",
    "LangGraph 是什么？",
    "文档分块有哪些策略？",
]


def _top(docs, k=3):
    return [{"source": d["source"], "score": d["score"]} for d in docs[:k]]


async def main():
    out = {"config": {
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "reranker": settings.rag_cross_encoder,
    }, "results": {}}

    # ---- 1) BM25-only ----
    bm25 = BM25Retriever(chunk_size=settings.rag_chunk_size,
                         chunk_overlap=settings.rag_chunk_overlap)
    bm25.ingest_directory(settings.knowledge_base_path)
    out["results"]["bm25"] = {
        q: _top(await bm25.search(q, settings.top_k)) for q in QUERIES
    }

    # ---- 2) hybrid 无重排 ----
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    dense = DenseRetriever(embedder=embedder, collection=settings.qdrant_collection,
                           url=settings.qdrant_url, local_path=settings.qdrant_path,
                           chunk_size=settings.rag_chunk_size,
                           chunk_overlap=settings.rag_chunk_overlap)
    await dense.ingest_directory(settings.knowledge_base_path, force=True)
    hybrid = HybridRetriever(dense=dense, bm25=bm25, top_k=settings.top_k)
    out["results"]["hybrid"] = {
        q: _top(await hybrid.search(q, settings.top_k)) for q in QUERIES
    }

    # ---- 3) hybrid + cross-encoder 重排 ----
    t0 = time.time()
    hybrid_rerank = HybridRetriever(dense=dense, bm25=bm25, top_k=settings.top_k,
                                    cross_encoder="BAAI/bge-reranker-base")
    out["results"]["hybrid_rerank"] = {}
    for q in QUERIES:
        out["results"]["hybrid_rerank"][q] = _top(await hybrid_rerank.search(q, settings.top_k))
    out["rerank_load_seconds"] = round(time.time() - t0, 1)

    # 写结果文件（UTF-8）
    with open("compare_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 打印摘要
    for q in QUERIES:
        print(f"\n=== {q} ===")
        for mode in ("bm25", "hybrid", "hybrid_rerank"):
            srcs = [f"{d['source']}({d['score']})" for d in out["results"][mode][q]]
            print(f"  [{mode:14s}] {srcs}")


if __name__ == "__main__":
    asyncio.run(main())
