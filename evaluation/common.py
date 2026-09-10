"""评测公共工具：数据集加载 + token 计数（统一口径）。

token 口径说明：本项目是中文售后场景，采用「中文单字 ≈ 1 token、英文/数字词 ≈ 1 token」
的近似（对标项目自实现的 _tokenize 粒度）。该口径用于对比「有记忆 vs 无记忆」的 prompt
长度下降比例，比例对 tokenizer 选择不敏感，结论稳定；绝对 token 数为估值，仅供量级参考。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_KB_DIR = _EVAL_DIR.parent / "data" / "knowledge_base"

_CJK_RE = re.compile(r"[一-鿿]")
_ASCII_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def count_tokens(text: str) -> int:
    """近似 token 计数：中文单字 1、英文/数字连续串 1，其余标点空白不计。"""
    if not text:
        return 0
    return len(_CJK_RE.findall(text)) + len(_ASCII_WORD_RE.findall(text))


def load_dataset() -> dict:
    with open(_EVAL_DIR / "eval_dataset.json", encoding="utf-8") as f:
        return json.load(f)


def questions() -> list[dict]:
    return load_dataset()["questions"]
