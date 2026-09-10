"""安全检查工具 — 复用 HELLO 的规则引擎，并包装成 LangChain @tool。

两类检测（确定性、离线、可测试）：
  1. 敏感词匹配 —— 内置默认词表 + .env 的 SAFETY_SENSITIVE_WORDS 追加
  2. PII 正则检测 —— 手机号、身份证号、银行卡号、邮箱

说明：这是「规则引擎示例」。生产可替换为基于 LLM 的安全审核（如 R-Judge）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

DEFAULT_SENSITIVE_WORDS: List[str] = [
    "炸弹制作",
    "制造炸药",
    "如何杀人",
    "购买枪支",
    "毒品配方",
    "自杀方法",
]

# PII 检测规则：标签 -> 正则
_PII_PATTERNS: List[tuple] = [
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("身份证号", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("银行卡号", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("邮箱地址", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]


class SafetyTool:
    """安全检查工具（同 HELLO，取即用）。

    在流水线中位于「生成之后」：命中敏感/PII 时拦截，不向用户返回风险文本。
    """

    name = "safety_check"
    description = "检测文本中的有害/敏感/泄露个人信息内容，返回是否安全及风险详情"

    def __init__(self, sensitive_words: List[str] | None = None):
        self.sensitive_words = list(
            set(DEFAULT_SENSITIVE_WORDS) | set(sensitive_words or [])
        )

    def _check_sensitive_words(self, text: str) -> List[str]:
        return [w for w in self.sensitive_words if w in text]

    def _check_pii(self, text: str) -> List[str]:
        return [label for label, pattern in _PII_PATTERNS if pattern.search(text)]

    async def execute(self, text: str) -> Dict[str, Any]:
        """执行安全检查，返回 {"safe", "risk_level", "risk_details"}。"""
        sensitive_hits = self._check_sensitive_words(text)
        pii_hits = self._check_pii(text)

        details: List[str] = []
        if sensitive_hits:
            details.append(f"命中敏感词: {', '.join(sensitive_hits)}")
        if pii_hits:
            details.append(f"疑似泄露个人信息: {', '.join(pii_hits)}")

        if sensitive_hits:
            risk_level = "high"
        elif pii_hits:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "safe": not details,
            "risk_level": risk_level,
            "risk_details": details,
        }


def build_safety_tool(sensitive_words: List[str] | None = None) -> SafetyTool:
    return SafetyTool(sensitive_words=sensitive_words)
