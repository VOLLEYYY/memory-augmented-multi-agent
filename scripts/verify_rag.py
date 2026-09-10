"""Day4 验证脚本：确认稠密检索真落地 + cross-encoder 开关。"""
import asyncio
import sys
import time

from app.config import settings
from app.core.rag import Embedder, build_retriever


async def main():
    print("=" * 60)
    print(f"embedding_model   = {settings.embedding_model}")
    print(f"embedding_device  = {settings.embedding_device}")
    print(f"rag_cross_encoder = {settings.rag_cross_encoder!r}")
    print(f"use_dense         = {settings.use_dense_retrieval}")
    print("=" * 60)

    # 1) Embedder 是否真能加载 sentence-transformers（不再哈希兜底）
    emb = Embedder(settings.embedding_model, settings.embedding_device)
    t0 = time.time()
    ok = emb.lazy_load()
    print(f"[embedder] lazy_load() -> {ok}  (耗时 {time.time()-t0:.1f}s)")
    if not ok:
        print("!! 稠密仍走哈希兜底，退出")
        return

    # 2) 构建检索器（会真摄入 qdrant）
    t0 = time.time()
    retriever, mode = await build_retriever(force_reload=True)
    print(f"[retriever] mode={mode}  type={type(retriever).__name__}  (耗时 {time.time()-t0:.1f}s)")

    # 3) 确认 dense 是否真写入（非 None）
    dense = getattr(retriever, "dense", None)
    bm25 = getattr(retriever, "bm25", None)
    print(f"[retriever] dense={type(dense).__name__ if dense else None} "
          f"dense.chunk_count={dense.chunk_count if dense else 'N/A'} "
          f"bm25.chunk_count={bm25.chunk_count if bm25 else 'N/A'}")

    # 4) 真实检索几条，看来源命中
    for q in ["什么是混合检索？", "LangGraph 是什么？", "RAG 评估有哪些指标？"]:
        docs = await retriever.search(q, 3)
        sources = [d["source"] for d in docs]
        print(f"\nQ: {q}")
        print(f"  sources={sources}")
        for d in docs[:2]:
            print(f"    - [{d['source']}] score={d['score']} :: {d['content'][:40]!r}")


if __name__ == "__main__":
    asyncio.run(main())
