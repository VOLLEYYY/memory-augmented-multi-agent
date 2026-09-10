"""语义记忆（长期）—— 用户事实/偏好，结构化存 SQLite + 简单检索。

与 episodic 的区别：
- episodic 是「某次对话发生了什么」（向量检索，时序 TTL）；
- semantic 是「这个用户是谁、偏好什么」（事实型，要求一致性、去重，长期保留）。

存储：SQLite 一表（stdlib sqlite3，零外部依赖，任何环境能跑）。
关键策略（面试追问点）：
- 去重/合并：新事实与已有事实相似度过高则合并；矛盾时「以新为准」+ 记录 old_value。
- 结构化：`key`（如 user.hobby）+ `value` + 更新次数，便于按类别管理。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import settings
from .models import MemoryItem, MemoryRetrieval, MemoryType

logger = logging.getLogger("agent.memory.semantic")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_memory (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    content TEXT,
    importance REAL DEFAULT 0.5,
    source TEXT DEFAULT 'user',
    created_at REAL,
    updated_at REAL,
    update_count INTEGER DEFAULT 1,
    old_value TEXT,
    metadata TEXT
);
"""

# 列名 -> 列序（与 _SCHEMA 严格对齐），便于按名取值，避免索引错位
_COLUMNS = [
    "id", "key", "value", "content", "importance", "source",
    "created_at", "updated_at", "update_count", "old_value", "metadata",
]
_COL_IDX = {name: i for i, name in enumerate(_COLUMNS)}


def _row_to_dict(row: tuple) -> Dict[str, Any]:
    return {name: row[i] for name, i in _COL_IDX.items()}


class SemanticMemory:
    """语义记忆：SQLite 持久化的用户事实/偏好。线程安全（单写锁）。"""

    def __init__(self, path: str | None = None):
        self.path = path or settings.sqlite_memory_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ---- 写入：去重 + 合并（以新为准）----
    async def upsert(self, key: str, value: str, source: str = "user",
                     importance: float = 0.5, content: str | None = None) -> MemoryItem:
        """写入一条语义记忆。

        若同 key 已存在且 value 不同，则「以新为准」并保留 old_value 供审计。
        """
        with self._lock:
            conn = self._connect()
            now = time.time()
            row = conn.execute(
                "SELECT value, update_count FROM semantic_memory WHERE key=?", (key,)
            ).fetchone()
            item_id = f"semantic_{abs(hash(key)) % (10**12)}"
            if row:
                old_value = row[0]
                update_count = row[1] + 1
                if old_value != value:
                    conn.execute(
                        """UPDATE semantic_memory SET value=?, old_value=?, update_count=?,
                           updated_at=?, content=?, importance=?, source=? WHERE key=?""",
                        (value, old_value, update_count, now, content or value,
                         importance, source, key),
                    )
            else:
                conn.execute(
                    """INSERT INTO semantic_memory
                       (id, key, value, content, importance, source, created_at, updated_at, update_count)
                       VALUES (?,?,?,?,?,?,?,?,1)""",
                    (item_id, key, value, content or value, importance, source, now, now),
                )
            conn.commit()
            item = MemoryItem(
                id=item_id, type=MemoryType.SEMANTIC, content=content or value,
                timestamp=now, importance=importance, source=source,
                metadata={"key": key, "update_count": (row[1] + 1) if row else 1},
            )
            return item

    # ---- 检索：按 key 精确 + 按内容词重叠打分 ----
    async def search(self, query: str, key: str | None = None, k: int = 5) -> MemoryRetrieval:
        with self._lock:
            conn = self._connect()
            if key:
                rows = conn.execute(
                    "SELECT * FROM semantic_memory WHERE key=?", (key,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM semantic_memory").fetchall()
        qterms = set(query.lower())
        scored: List[tuple] = []
        for r in rows:
            d = _row_to_dict(r)
            content = d["content"] or d["value"] or ""
            overlap = len(qterms & set(content.lower()))
            score = overlap + float(d["importance"] or 0.5)
            if overlap > 0 or key:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)

        items = []
        for _, d in scored[:k]:
            items.append(
                MemoryItem(
                    id=d["id"], type=MemoryType.SEMANTIC,
                    content=d["content"] or d["value"] or "",
                    timestamp=float(d["created_at"] or time.time()),
                    importance=float(d["importance"] or 0.5),
                    source=d["source"] or "user",
                    metadata={
                        "key": d["key"], "value": d["value"],
                        "old_value": d["old_value"], "update_count": d["update_count"],
                    },
                )
            )
        return MemoryRetrieval(items=items, hit=bool(items))

    # ---- 列出所有语义记忆（调试/前端展示）----
    async def all_items(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute("SELECT * FROM semantic_memory").fetchall()
        return [
            {
                "id": _row_to_dict(r)["id"], "key": _row_to_dict(r)["key"],
                "value": _row_to_dict(r)["value"], "content": _row_to_dict(r)["content"],
                "importance": _row_to_dict(r)["importance"], "source": _row_to_dict(r)["source"],
                "update_count": _row_to_dict(r)["update_count"], "old_value": _row_to_dict(r)["old_value"],
            }
            for r in rows
        ]


_instance: Optional[SemanticMemory] = None


def get_semantic_memory() -> SemanticMemory:
    global _instance
    if _instance is None:
        _instance = SemanticMemory()
    return _instance
