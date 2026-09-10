"""记忆数据模型。

记忆分层（对齐 hello-agents ch8 + 本项目落地）：
- WORKING   工作记忆：当前会话（LangGraph checkpointer 提供，不落 Qdrant）
- EPISODIC  情景记忆：跨会话的对话片段，向量化存 Qdrant
- SEMANTIC  语义记忆：用户事实/偏好，结构化存 SQLite + 可选向量
- PROCEDURAL 程序记忆：学会的做法/工作流（D7 后置）

设计要点：
- MemoryItem 是记忆的最小单元，各层共用同一结构，只靠 `type` 区分。
- `importance` 影响记忆权重与遗忘优先级；`source` 记录来源（如 "qa"/"user"）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


def _now() -> float:
    return time.time()


@dataclass
class MemoryItem:
    """一条记忆。

    Attributes:
        id:      唯一 id（生成时用时间戳+随机，可用前端 uuid 代替）
        type:    记忆类型（working/episodic/semantic/procedural）
        content: 记忆正文（文本）
        timestamp: 写入时间（epoch 秒），用于时序检索 / TTL 淘汰
        importance: 重要性 0~1，影响检索排序与固化保留
        source:  记忆来源（"user" / "qa" / "tool" / "consolidator"）
        metadata: 扩展字段（用户 id、会话 id、标签等）
    """

    id: str
    type: MemoryType
    content: str
    timestamp: float = field(default_factory=_now)
    importance: float = 0.5
    source: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(
            id=data["id"],
            type=MemoryType(data.get("type", "episodic")),
            content=data.get("content", ""),
            timestamp=float(data.get("timestamp", _now())),
            importance=float(data.get("importance", 0.5)),
            source=data.get("source", "user"),
            metadata=data.get("metadata", {}) or {},
        )


@dataclass
class MemoryRetrieval:
    """记忆检索结果：命中的记忆 + 是否命中（供上层判断要不要把记忆注入上下文）。"""

    items: list[MemoryItem] = field(default_factory=list)
    hit: bool = False

    def as_content(self) -> str:
        """拼接为可注入系统提示的文本。"""
        if not self.items:
            return ""
        return "\n\n".join(
            f"[记忆·{it.type.value}·{it.source} · 重要{it.importance:.1f}]\n{it.content}"
            for it in self.items
        )


@dataclass
class ConsolidationResult:
    """固化结果：保留的摘要 + 被改写/删除的条目数。"""

    kept_memories: list[MemoryItem] = field(default_factory=list)
    merged_count: int = 0
    dropped_count: int = 0
    summary: str = ""
