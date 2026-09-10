"""评测指标工具函数 —— 全部离线、确定性，不依赖外部 LLM API。

指标清单：
1. recall_at_k / ndcg_at_k —— 文档级检索指标（二值相关标签：相关=1，不相关=0）。
2. estimate_tokens —— 粗略但自洽的 token 估计（中文按字、英文/数字按词），
   用于 A/B 相对对比（同一估计器两边都用，比值可信；绝对值仅参考量级）。
3. intent_accuracy —— 意图分流准确率（预测 label vs 金标 label 的命中率）。

为什么「文档级」而非「chunk 级」：本项目知识库按「一篇售后政策 = 一个 source 文件」组织，
用户真正关心的是「有没有找到对的那篇政策」，故按 source（文件名）判定命中，更贴近业务语义。
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence

# ---- token 估计：中文按字、英文/数字按词。DeepSeek BPE 对中文约 1 token/字，量级接近。 ----
_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    """粗略但自洽的 token 估计。

    只用于 A/B 相对对比（量化 2 两边用同一估计器，比值/下降率可信）；
    绝对值是量级参考，非精确 token 数。
    """
    if not text:
        return 0
    return len(_CJK_RE.findall(text)) + len(_WORD_RE.findall(text))


# ---- 检索指标：source（文件名）级，二值相关 ----
def _sources(results: Sequence[Dict[str, Any]]) -> List[str]:
    return [r.get("source", "") for r in results]


def recall_at_k(results: Sequence[Dict[str, Any]], relevant: Sequence[str], k: int) -> float:
    """Recall@k：top-k 结果里命中了多少个相关文档（按 source 去重）÷ 相关文档总数。"""
    if not relevant:
        return 1.0
    retrieved = set(_sources(results[:k]))
    hits = sum(1 for r in relevant if r in retrieved)
    return hits / len(relevant)


def ndcg_at_k(results: Sequence[Dict[str, Any]], relevant: Sequence[str], k: int) -> float:
    """NDCG@k（文档级、二值相关）：相关文档排得越靠前，得分越高，取值 [0,1]。

    DCG = Σ rel_i / log2(i+2)，IDCG 用理想排序（所有相关文档都排在最前）。
    注意：同一 source 文件可能被切成多个 chunk 进入 top-k，故先按 source 去重（保留首次出现），
    否则同一相关文档会被重复计入、导致 NDCG > 1。
    """
    if not relevant:
        return 1.0
    rel = set(relevant)
    seen: set = set()
    dedup: List[str] = []
    for r in results[:k]:
        s = r.get("source", "")
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    dcg = 0.0
    for i, s in enumerate(dedup):
        if s in rel:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


# ---- 意图分流准确率 ----
def accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    """整体准确率。"""
    if not golds:
        return 0.0
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(golds)


def per_class_accuracy(preds: Sequence[str], golds: Sequence[str], classes: Sequence[str]):
    """每个类别的准确率 dict。"""
    out: Dict[str, Dict[str, float]] = {}
    for c in classes:
        idx = [i for i, g in enumerate(golds) if g == c]
        if not idx:
            out[c] = {"total": 0, "correct": 0, "acc": None}
            continue
        correct = sum(1 for i in idx if preds[i] == c)
        out[c] = {"total": len(idx), "correct": correct, "acc": correct / len(idx)}
    return out


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
