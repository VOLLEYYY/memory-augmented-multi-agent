"""重灌售后知识库 + 验证切片策略（Day 6 后 · 知识库升级）。

做的事：
  1. 列出 data/knowledge_base/*.md，统计每篇字数，标记长文档（体现「原材料要长」）。
  2. 用 rag._split 对最长几篇做「切片预览」，展示 512 字符 + 50 overlap 如何切开长文。
  3. build_retriever(force_reload=True) 强制重灌 Qdrant（删旧 collection 重建）。
  4. 报告 dense/bm25 chunk 数，并检索售后问题确认命中售后文档（技术文档已隔离）。

跑法（在项目根目录下）：PYTHONPATH=. python scripts/reindex_aftersales_kb.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import settings
from app.core.rag import _split, build_retriever

KB = Path(settings.knowledge_base_path)


def preview_splits(name: str, text: str) -> None:
    chunks = _split(text, settings.rag_chunk_size, settings.rag_chunk_overlap)
    print(f"\n  【切片预览】{name}（全文 {len(text)} 字 → {len(chunks)} 个 chunk）")
    # 展示前 2 个 chunk 的「尾/头」，看 overlap 是否生效
    for i in range(min(2, len(chunks))):
        c = chunks[i]
        print(f"    chunk[{i}]  {c[:36]!r} … {c[-12:]!r}")
    if len(chunks) >= 2:
        head_next = chunks[1][:50]
        print(f"    ↑ chunk[1] 开头 {head_next!r}  （与 chunk[0] 尾部的 {settings.rag_chunk_overlap} 字重叠）")


async def main() -> None:
    print("=" * 64)
    print(f"knowledge_base_path = {settings.knowledge_base_path}")
    print(f"rag_chunk_size      = {settings.rag_chunk_size}")
    print(f"rag_chunk_overlap   = {settings.rag_chunk_overlap}")
    print("=" * 64)

    # ---- 1) 字数统计，找长文档 ----
    files = sorted(KB.glob("*.md"))
    print(f"\n[1] 售后知识库共 {len(files)} 篇 md，字数统计：")
    meta = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        meta.append((f.name, len(text), text))
    meta.sort(key=lambda x: -x[1])
    for name, n, _ in meta:
        flag = "  ← 长文" if n >= 2000 else ""
        print(f"    {name:52s} {n:5d} 字{flag}")

    # ---- 2) 对最长 3 篇做切片预览 ----
    print("\n[2] 最长文档的切片预览（体现 chunk_size + overlap）：")
    for name, n, text in meta[:3]:
        preview_splits(name, text)

    # ---- 3) 强制重灌 ----
    print("\n[3] 重灌 Qdrant（force_reload=True，删旧 collection 重建）……")
    t0 = asyncio.get_event_loop().time()
    retriever, mode = await build_retriever(force_reload=True)
    dense = getattr(retriever, "dense", None)
    bm25 = getattr(retriever, "bm25", None)
    print(f"    mode={mode}  dense.chunk_count={dense.chunk_count if dense else 'N/A'}  "
          f"bm25.chunk_count={bm25.chunk_count if bm25 else 'N/A'}")

    # ---- 4) 售后问题检索验证 ----
    print("\n[4] 售后问题检索验证（应命中售后文档，而非技术文档）：")
    for q in ["智能门锁保修多久", "七天无理由退货怎么退", "产品修两次还坏能退吗"]:
        docs = await retriever.search(q, 3)
        sources = [d["source"] for d in docs]
        print(f"    Q: {q}")
        print(f"       sources = {sources}")


if __name__ == "__main__":
    asyncio.run(main())
