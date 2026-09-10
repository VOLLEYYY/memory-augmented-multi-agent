"""FastAPI 入口。

已落地 HELLO 的两个待优化点：
1. `@app.on_event` → `lifespan` 上下文管理器（不再用废弃 API）。
2. 统一异常处理器 + request_id 中间件（不再在 routes 里 try/except 抛裸 500）。

lifespan 里做「预热」：构建 LangGraph、初始化检索器 —— 把重一次性处理放启动，
避免首个请求冷加载。任何环节失败都降级（graph=None 时走直连降级路径）。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.core.rate_limit import limiter
from app.config import settings
from app.core import graph
from app.core.observability import get_request_id, log_event, set_request_id, setup_logging
from app.core.services import get_services

logger = logging.getLogger("agent.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("🚀 %s 启动中 ...", settings.app_name)

    # 预热：构建 LangGraph + 初始化检索器（失败降级，不阻塞启动）
    try:
        await graph.get_graph()
    except Exception as e:
        logger.warning("启动时 graph 预热失败（%s）", e)
    try:
        await get_services().ensure_retriever()
        logger.info("检索器就绪：mode=%s", get_services().retriever_mode)
    except Exception as e:
        logger.warning("启动时检索器预热失败（%s），将按需构建", e)

    yield
    logger.info("👋 关闭")
    # 收尾：关闭记忆存储 + checkpointer sqlite 连接
    try:
        from app.memory.semantic import get_semantic_memory
        get_semantic_memory().close()
    except Exception:
        pass
    try:
        await graph.close_checkpointer()
    except Exception:
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    # 限流：挂 slowapi 的 limiter + 429 异常处理
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # request_id 中间件：每请求生成一个 id，贯穿日志与返回
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or set_request_id()
        set_request_id(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    # 统一异常处理器：返回结构化 JSON（对外笼统，对内记日志）
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("未处理异常 rid=%s", get_request_id())
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "服务内部错误", "request_id": get_request_id()},
        )

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
