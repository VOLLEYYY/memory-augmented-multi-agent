"""工作记忆（短期）—— 当前会话的对话历史。

实现策略：
- 主路径：复用 LangGraph 的 `checkpointer`（InMemorySaver -> SqliteSaver 一行换），
  LangGraph 按 `thread_id` 持久化 `messages`，天然就是工作记忆。
- 本模块提供对「最近 N 轮」的读取与截断工具，与 graph 的 share state 解耦，
  方便在记忆子 Agent 里按需取用（不直接依赖 graph 内部结构）。

注意：工作记忆 ≠ 跨会话长期记忆（那个在 episodic/semantic）。这里只是「当前线程最近对话」。
"""
from __future__ import annotations

from typing import Any, Dict, List

# 默认：工作记忆里最多保留多少条消息，超出按「保头+保尾」截断，避免上下文膨胀。
_DEFAULT_KEEP = 20


def extract_recent_messages(
    messages: List[Dict[str, Any]], keep: int = _DEFAULT_KEEP
) -> List[Dict[str, Any]]:
    """从消息列表截取「最近 keep 条」，保头 1 条 + 尾部，防止丢失 system 说明。"""
    if not messages:
        return []
    if len(messages) <= keep:
        return messages
    return messages[:1] + messages[-(keep - 1):]


def recent_user_content(messages: List[Dict[str, Any]]) -> str:
    """取最近一条 user 消息正文（供记忆子 Agent 快速定位当前问题）。"""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


class WorkingMemory:
    """工作记忆门面。

    说明：真正的持久化由 LangGraph checkpointer 承担；这里提供统一接口，
      后续若想换 Redis/Mongo 存工作记忆，只需改这个类，graph 不变。
    """

    def __init__(self, keep: int = _DEFAULT_KEEP):
        self.keep = keep

    def load(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """返回当前线程最近的有效消息（已做截断）。"""
        return extract_recent_messages(messages, self.keep)

    def current_question(self, messages: List[Dict[str, Any]]) -> str:
        return recent_user_content(messages)

    def summarize(self, messages: List[Dict[str, Any]]) -> str:
        """工作记忆轻量摘要：仅取最近若干条 user/assistant 的对偶，供注入。"""
        trimmed = self.load(messages)
        pairs = []
        pending = ""
        for m in trimmed:
            if m.get("role") == "user":
                pending = str(m.get("content", ""))[:100]
            elif m.get("role") == "assistant" and pending:
                pairs.append(f"Q:{pending} … A:{str(m.get('content', ''))[:80]}")
                pending = ""
        return "\n".join(pairs) if pairs else ""
