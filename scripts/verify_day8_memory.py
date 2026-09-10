"""Day8 验证脚本：跨会话「工作记忆」+ 跨进程(重启)持久化 + 跨 thread 隔离。

每次调用 = 一个独立进程（等价于服务重启后重新连接 memory.db 的 checkpointer）。
只验证「工作记忆」链路（checkpointer 的 messages 跨轮累积 + memory 节点注入），
episodic 长期记忆的跨进程持久化已在 Day6 的 verify_episodic.py 验证过，这里用
_InMemoryEpisodic 占位以避开 embedding 加载，让验证秒级完成、不依赖网络/key。

用法（三次独立进程）：
  python verify_day8_memory.py "我买的智能门锁A100保修期多久" user-1   # 第1轮，写入工作记忆
  python verify_day8_memory.py "那我刚才说的门锁想换货怎么操作" user-1  # 第2轮=重启后，应注入上一轮 Q:A
  python verify_day8_memory.py "那我刚才说的门锁想换货怎么操作" user-2  # 跨 thread，应无上一轮上下文
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.graph import run_agent
from app.core.services import get_services
from app.memory.episodic import _InMemoryEpisodic
from app.memory.semantic import SemanticMemory


class _FakeRetriever:
    async def search(self, query: str, top_k: int):
        return []


async def _fake_ensure_retriever(*args, **kwargs):
    return _FakeRetriever()


async def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    question, thread_id = sys.argv[1], sys.argv[2]

    # 把单例 services 换成「无重依赖」版本：不加载 embedding、不调 LLM、不检索知识库
    services = get_services()
    services.episodic = _InMemoryEpisodic()
    services.semantic = SemanticMemory(path=os.path.join(tempfile.mkdtemp(), "sem.db"))
    services.model_a = lambda: None
    services.model_b = lambda: None
    services.ensure_retriever = _fake_ensure_retriever

    result = await run_agent(question, request_id="", thread_id=thread_id)

    print("=" * 64)
    print(f"thread_id      = {thread_id}")
    print(f"question       = {question}")
    print(f"memory_hit     = {result.get('memory_hit')}")
    print(f"memory_context =")
    print(result.get("memory_context", "（空）") or "（空）")
    msgs = result.get("messages") or []
    print(f"messages roles = {[m.get('role') for m in msgs]}")
    print(f"messages count = {len(msgs)}")


if __name__ == "__main__":
    asyncio.run(main())
