"""HTTP 路由层 —— 只做「收发请求」，业务编排交给 core/graph.py（分层思想同上个项目）。

端点：
- GET  /api/v1/health                      健康检查
- GET  /api/v1/status                      graph/检索器状态（调试/面试展示）
- POST /api/v1/ask                         问答（可流式）
- POST /api/v1/memory/upsert              写入一条语义记忆（用户偏好）
- POST /api/v1/memory/consolidate         触发记忆固化
- GET  /api/v1/memory                     列出语义记忆
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import settings
from ..core import graph as graph_mod
from ..core.rate_limit import limiter
from ..core.graph import graph_status, run_agent, run_agent_stream
from ..core.observability import get_request_id, log_event, set_request_id
from ..core.services import get_services
from ..memory.consolidator import get_consolidator
from ..models.schemas import (
    AskRequest,
    AskResponse,
    ConsolidateRequest,
    ConsolidateResponse,
    GraphStatusResponse,
    MemoryUpsertRequest,
)

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/status", response_model=GraphStatusResponse)
async def status():
    gs = graph_status()
    # 检索器模式：惰性构建一次
    try:
        services = get_services()
        await services.ensure_retriever()
        retriever_mode = services.retriever_mode
    except Exception:
        retriever_mode = "unavailable"
    return GraphStatusResponse(compiled=gs["compiled"], error=gs["error"], retriever_mode=retriever_mode)


@router.post("/ask")
@limiter.limit(settings.rate_limit_ask)
async def ask(req: AskRequest, request: Request):
    rid = set_request_id()
    log_event("ask_start", question=req.question, streaming=req.streaming,
              thread_id=req.thread_id)

    if req.streaming:
        async def gen():
            try:
                async for event in run_agent_stream(req.question, rid, req.thread_id):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        result = await run_agent(req.question, rid, req.thread_id)
    except Exception as e:
        log_event("ask_error", error=str(e))
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

    safety = result.get("safety") or {}
    resp = AskResponse(
        success=True,
        answer=result.get("final_answer") or result.get("answer", ""),
        original_answer=result.get("answer", ""),
        sources=result.get("sources", []) or [],
        memory_hit=bool(result.get("memory_hit", False)),
        memory_context=result.get("memory_context", ""),
        review_score=(result.get("review") or {}).get("overall"),
        review_comment=result.get("review_comment", ""),
        need_revision=bool(result.get("need_revision", False)),
        safety_passed=safety.get("safe", True),
        route=result.get("route", ""),
        request_id=rid,
    )
    log_event("ask_done", answer_len=len(resp.answer), memory_hit=resp.memory_hit)
    return resp


@router.post("/memory/upsert")
async def memory_upsert(req: MemoryUpsertRequest):
    set_request_id()
    from app.memory.semantic import get_semantic_memory
    item = await get_semantic_memory().upsert(req.key, req.value, source=req.source)
    return {"success": True, "item": item.to_dict()}


@router.post("/memory/consolidate", response_model=ConsolidateResponse)
async def memory_consolidate(req: ConsolidateRequest):
    set_request_id()
    result = await get_consolidator().consolidate()
    return ConsolidateResponse(
        success=True,
        summary=result.summary,
        merged_count=result.merged_count,
        kept_count=len(result.kept_memories),
    )


@router.get("/memory")
async def list_memory():
    from app.memory.semantic import get_semantic_memory
    items = await get_semantic_memory().all_items()
    return {"success": True, "count": len(items), "items": items}
