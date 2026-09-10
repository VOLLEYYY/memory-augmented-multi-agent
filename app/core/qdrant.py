"""共享 Qdrant 客户端 —— 全局唯一，避免本地嵌入式模式的并发冲突。

坑（A 实测抓到）：Qdrant 本地模式（`QdrantClient(path=...)`）**一个文件夹只能被一个
客户端实例打开**。如果 RAG 检索器和 episodic 记忆各自 new 一个本地客户端，就会报
"Storage folder ... is already accessed by another instance"。

修复：全局维护**唯一**客户端（单例），RAG DenseRetriever 与 EpisodicMemory 都复用它，
不同 collection 共用同一个持久化文件，互不冲突。
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from ..config import settings

logger = logging.getLogger("agent.qdrant")

_client: Optional[Any] = None
_lock = threading.Lock()


def get_qdrant_client() -> Any:
    """获取全局唯一 Qdrant 客户端（惰性创建，线程安全）。

    连接模式（qdrant_url 优先，其次 qdrant_path，最后内存）：
      - url="http://localhost:6333"  服务端模式（Docker/生产）
      - path="./qdrant_data"         本地嵌入式模式（无需 Docker）
      - 二者皆空                      内存模式（测试，重启即失）
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                from qdrant_client import QdrantClient
                if settings.qdrant_url:
                    _client = QdrantClient(url=settings.qdrant_url)
                    logger.info("Qdrant 服务端模式：%s", settings.qdrant_url)
                elif settings.qdrant_path:
                    _client = QdrantClient(path=settings.qdrant_path)
                    logger.info("Qdrant 本地模式：%s（全局唯一客户端）", settings.qdrant_path)
                else:
                    _client = QdrantClient(":memory:")
                    logger.info("Qdrant 内存模式（测试）")
    return _client


def reset_client() -> None:
    """测试/重载用：重置单例。"""
    global _client
    _client = None
