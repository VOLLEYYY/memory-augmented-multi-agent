"""服务容器 — 集中持有各子 Agent 共享的组件（检索器/模型/记忆/安全）。

为什么单独拆一个模块：
- LangGraph 的节点函数、子 Agent 模块都要用到这些组件；若把它们定义在 graph.py 里，
  会造成 graph ↔ subagents 互相 import 的环。
- 放进独立容器既避免循环依赖，又便于测试（测试里可以注入 fake 组件）。

构造通过 `build_services()` 惰性完成（真用到才构建，且任何环节失败都降级，不阻塞启动）。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import settings
from ..memory.episodic import EpisodicMemory, get_episodic_memory
from ..memory.semantic import SemanticMemory, get_semantic_memory
from ..memory.consolidator import Consolidator, get_consolidator
from ..tools.safety_tool import SafetyTool
from .llm import LLMClient, get_chat_model
from .rag import build_retriever

logger = logging.getLogger("agent.services")


class Services:
    """所有可被注入的子 Agent 依赖。属性为「惰性构建 + 可降级」。"""

    def __init__(self) -> None:
        self.retriever: Optional[Any] = None
        self.retriever_mode: str = "hybrid"
        self.episodic: EpisodicMemory = get_episodic_memory()
        self.semantic: SemanticMemory = get_semantic_memory()
        self.consolidator: Consolidator = get_consolidator()
        self.safety: SafetyTool = SafetyTool(
            sensitive_words=settings.get_safety_sensitive_words_list()
        )
        self.llm_a: Optional[LLMClient] = None
        self.llm_b: Optional[LLMClient] = None
        self.llm_a_langchain: Optional[Any] = None

    # ---- 模型：生成 A / 评审 B（懒加载，失败返回 None，由节点降级）----
    def model_a(self) -> Optional[LLMClient]:
        if self.llm_a is None:
            try:
                self.llm_a = LLMClient(role="a")
            except Exception as e:
                logger.warning("模型A 不可用（%s），将降级", e)
                self.llm_a = None  # 保持 None，后续节点判空降级
        return self._or_none(self.llm_a)

    def model_b(self) -> Optional[LLMClient]:
        if self.llm_b is None:
            try:
                self.llm_b = LLMClient(role="b")
            except Exception as e:
                logger.warning("模型B 不可用（%s），将降级", e)
                self.llm_b = None
        return self._or_none(self.llm_b)

    @staticmethod
    def _or_none(obj):
        return obj if obj is not None else None

    # ---- 检索器：惰性构建，失败降级 ----
    async def ensure_retriever(self, force_reload: bool = False) -> Any:
        if self.retriever is None or force_reload:
            self.retriever, self.retriever_mode = await build_retriever(force_reload)
        return self.retriever


_singleton: Optional[Services] = None


def get_services() -> Services:
    """全局单例（与外部 service 生命周期一致，测试可注入替换）。"""
    global _singleton
    if _singleton is None:
        _singleton = Services()
    return _singleton
