"""Day11 熔断复验：LLMClient 连续失败 → 熔断 → 快速失败。

用错误 key 构造 LLMClient（模拟外部 LLM 持续故障），连续 ainvoke，观察：
  前 N 次（N=llm_circuit_failures）真调外部、抛认证错（OpenAIAuthenticationError）；
  第 N+1 次起抛 CircuitBreakerError（熔断打开，快速失败，不再真调外部）。
"""
from __future__ import annotations

import os

os.environ["MODEL_A_API_KEY"] = "wrong-key-for-circuit-test"  # 必须在 import app 之前设置

import asyncio
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.circuit_breaker import CircuitBreakerError, CircuitState  # noqa: E402
from app.core.llm import LLMClient  # noqa: E402


async def main() -> None:
    llm = LLMClient(role="a")
    thr = llm._breaker.failure_threshold
    print(f"breaker 初始状态={llm._breaker.state.value}，阈值={thr}")
    opened_at = None
    for i in range(1, thr + 3):
        try:
            await llm.ainvoke([{"role": "user", "content": "hi"}])
            print(f"第{i}次: 意外成功")
        except CircuitBreakerError as e:
            print(f"第{i}次: CircuitBreakerError（熔断快速失败，不真调外部） state={llm._breaker.state.value}")
            if opened_at is None:
                opened_at = i
        except Exception as e:
            print(f"第{i}次: {type(e).__name__}（真调外部失败） state={llm._breaker.state.value}")

    ok = opened_at == thr + 1  # 第 thr+1 次开始快速失败
    print("=" * 64)
    print(("✅" if ok else "❌") + f" 熔断在第 {thr}+1 次触发（前 {thr} 次真调失败，之后快速失败）")


if __name__ == "__main__":
    asyncio.run(main())
