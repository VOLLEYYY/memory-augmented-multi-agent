"""API 请求/响应数据模型（Pydantic）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    thread_id: str = Field("", description="会话 id：同一 session 复用同一 thread，工作记忆跨轮保留")
    streaming: bool = Field(False, description="是否流式返回")


class SourceInfo(BaseModel):
    source: Optional[str] = None
    content: str = ""
    score: Optional[float] = None


class AskResponse(BaseModel):
    success: bool
    answer: str
    original_answer: str = ""
    sources: List[str] = []
    memory_hit: bool = False
    memory_context: str = ""
    review_score: Optional[float] = None
    review_comment: str = ""
    need_revision: bool = False
    safety_passed: bool = True
    route: str = ""
    request_id: str = ""


class MemoryUpsertRequest(BaseModel):
    key: str
    value: str
    source: str = "user"


class ConsolidateRequest(BaseModel):
    force: bool = Field(False, description="True 时忽略计数直接固化")


class ConsolidateResponse(BaseModel):
    success: bool
    summary: str
    merged_count: int = 0
    kept_count: int = 0


class GraphStatusResponse(BaseModel):
    compiled: bool
    error: Optional[str] = None
    retriever_mode: str = "hybrid"
