"""LLM 调用封装 — 基于 LangChain ChatOpenAI（OpenAI 兼容协议）。

设计要点（对齐 HELLO 的"优雅降级"哲学 + LangChain 生态）：
1. 用 `langchain_openai.ChatOpenAI` 作为统一模型接口 —— 可无缝切换 DeepSeek / Ollama /
   vLLM 等所有 OpenAI 兼容后端（只需改 .env 的 base_url/name）。
2. 生成模型（角色 "a"）与评审模型（角色 "b"）分离，实现「双模型物理隔离」。
3. 重试交给 ChatOpenAI 内置 `max_retries`（指数退避由 SDK 处理）；上层再包一层
   `asyncio.wait_for` 防止整体失控。失败时抛异常，由编排层降级。

为什么用 LangChain 而不是裸 openai SDK：
- 后续 D10 Function Calling 直接用 `model.bind_tools(...)`，与 LangGraph 的 ToolNode 天然打通；
- 面试口径："底层协议我懂（HELLO 手写过），编排层用 LangChain/LangGraph 省状态管理"。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, List

from ..config import settings
from .circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger("agent.llm")


def _resolve(role: str) -> tuple[str, str, str]:
    """按 role 返回 (base_url, api_key, model_name)。

    role: "a" 生成模型 | "b" 评审模型。空 base_url 视为未配置（交由上层降级）。
    """
    if role == "a":
        return settings.model_a_base_url, settings.model_a_api_key, settings.model_a_name
    return settings.model_b_base_url, settings.model_b_api_key, settings.model_b_name


def get_chat_model(role: str = "a", temperature: float | None = None) -> Any:
    """构建一个 LangChain ChatOpenAI 实例。依赖缺失/未配置时抛异常，由调用方降级。

    Returns:
        一个 langchain_openai.ChatOpenAI 实例（带 max_retries）。
    """
    from langchain_openai import ChatOpenAI  # 延迟导入：未安装则抛 ImportError

    base_url, api_key, model = _resolve(role)
    if not base_url:
        raise ValueError("LLM 未配置 base_url（请在 .env 设置 MODEL_A_BASE_URL）")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key or "not-needed",
        temperature=settings.llm_temperature if temperature is None else temperature,
        timeout=settings.llm_timeout,
        max_retries=2,
    )


class LLMClient:
    """封装一次/流式文本生成，带超时保护与降级提示。

    业务代码只依赖 `ainvoke` / `astream` 两个方法；内部用 LangChain 模型，失败抛异常。
    """

    def __init__(self, role: str = "a", temperature: float | None = None):
        self.role = role
        self._model: Any = None
        # 熔断器：外部 LLM 连续失败 N 次后快速失败，冷却后半开试探（防雪崩）
        self._breaker = CircuitBreaker(
            failure_threshold=settings.llm_circuit_failures,
            recovery_timeout=settings.llm_circuit_recovery,
        )

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = get_chat_model(self.role)
        return self._model

    async def ainvoke(
        self, messages: list[dict[str, Any]], temperature: float | None = None
    ) -> str:
        """一次性生成，返回文本。带整体超时保护 + 熔断。

        messages 形如 [{"role": "user", "content": "..."}]。
        """
        # 熔断打开 → 快速失败（不真调外部），上层据此降级
        if not self._breaker.allow_request():
            raise CircuitBreakerError(
                f"模型 {self.role} 已熔断，快速失败"
                f"（连续 {self._breaker.failure_threshold} 次失败触发）"
            )
        model = self._get_model()
        if temperature is not None:
            model = model.bind(temperature=temperature)
        try:
            resp = await asyncio.wait_for(
                model.ainvoke(messages), timeout=settings.llm_timeout
            )
        except Exception:
            self._breaker.record_failure()
            raise
        self._breaker.record_success()
        return str(resp.content or "")

    async def astream(
        self, messages: list[dict[str, Any]]
    ) -> AsyncIterator[str]:
        """流式生成，逐 token 产出文本（供 SSE）。"""
        model = self._get_model()
        async for chunk in model.astream(messages):
            text = chunk.content
            if isinstance(text, str) and text:
                yield text
            elif isinstance(text, list):  # 某些后端 content 是 list[dict] 结构
                for part in text:
                    if isinstance(part, dict) and part.get("text"):
                        yield part["text"]
