"""量化 3：检索指标 Recall@k / NDCG@k —— BM25 vs 稠密 vs 混合。

在售后知识库上，用带 ground-truth source 的评测集，比较三种检索器的排序质量：
- BM25（稀疏，零依赖）
- Dense（bge-small-zh-v1.5 + Qdrant，稠密向量）
- Hybrid（BM25 + 稠密 → RRF 融合，默认不含 cross-encoder 重排）

指标口径：
- Recall@k：正确来源是否出现在 top-k（衡量「有没有召回对」）。
- NDCG@k：正确来源的排名位置（衡量「排得靠不靠前」），单正确答案时 NDCG@k = 1/log2(rank+2)。
"""
from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.common import questions  # noqa: E402

_KB_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"


def _dcg_at(rank_of_gt: int, k: int) -> float:
    """单正确答案的 NDCG@k：rank_of_gt 为正确来源在 top-k 中的 0 基位置，未命中返回 0。"""
    if rank_of_gt < 0 or rank_of_gt >= k:
        return 0.0
    return 1.0 / math.log2(rank_of_gt + 2)  # DCG=1/log2(pos+2)，IDCG=1（理想排第 1）


async def _evaluate_retriever(retriever, queries, k=5):
    """返回 {recall@1, recall@3, recall@5, ndcg@3, ndcg@5} 的平均值。"""
    recalls = {1: [], 3: [], 5: []}
    ndcgs = {3: [], 5: []}
    for item in queries:
        gt = item["kb_source"]
        try:
            docs = await retriever.search(item["question"], k)
        except Exception:
            docs = []
        sources = [d.get("source") for d in docs]
        for kk in recalls:
            recalls[kk].append(1.0 if gt in sources[:kk] else 0.0)
        for kk in ndcgs:
            pos = sources.index(gt) if gt in sources[:kk] else -1
            ndcgs[kk].append(_dcg_at(pos, kk))
    out = {"n": len(queries)}
    for kk in recalls:
        out[f"recall@{kk}"] = round(sum(recalls[kk]) / len(recalls[kk]), 3) if recalls[kk] else 0.0
    for kk in ndcgs:
        out[f"ndcg@{kk}"] = round(sum(ndcgs[kk]) / len(ndcgs[kk]), 3) if ndcgs[kk] else 0.0
    return out


async def main() -> dict:
    from app.core.rag import BM25Retriever, DenseRetriever, HybridRetriever

    qs = [q for q in questions() if q["kb_source"]]
    print(f"评测集：{len(qs)} 条需知识库/需评审问题（含 ground-truth source）")
    print("=" * 70)

    # 1. BM25（零依赖，任何环境可跑）
    bm25 = BM25Retriever()
    n_chunk = bm25.ingest_directory(str(_KB_DIR))
    bm25_res = await _evaluate_retriever(bm25, qs)
    print(f"[BM25]        chunk={n_chunk}  {bm25_res}")

    # 2. Dense（Qdrant + bge 稠密）
    dense_res = None
    try:
        dense = DenseRetriever(collection="eval_kb")
        await dense.ingest_directory(str(_KB_DIR), force=True)
        dense_res = await _evaluate_retriever(dense, qs)
        print(f"[Dense]       {dense_res}")
    except Exception as e:
        print(f"[Dense]       不可用（{e.__class__.__name__}: {e}），跳过")

    # 3. Hybrid（RRF 融合）
    hybrid = HybridRetriever(dense=dense if dense_res else None, bm25=bm25)
    hybrid_res = await _evaluate_retriever(hybrid, qs)
    print(f"[Hybrid]      {hybrid_res}")

    print("=" * 70)
    result = {"bm25": bm25_res, "dense": dense_res, "hybrid": hybrid_res, "n_chunk": n_chunk}
    return result


if __name__ == "__main__":
    asyncio.run(main())
