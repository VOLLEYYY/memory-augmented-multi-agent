"""可观测性 — request_id 贯穿 + 结构化日志 + Langfuse 门面（可选）。

对应 HELLO 复盘待优化点「日志未结构化、无 request_id」的落地：
1. 用 ContextVar 在「一次请求」内传递 request_id，贯穿 检索 → 生成 → 评审。
2. 提供 logging.Filter，把 request_id 注入每条日志记录的 `extra`。
3. 提供快捷函数 `set_request_id` / `get_request_id` / `log_event`，供子 Agent 打点。
4. Langfuse 门面（可选）：未装依赖/未配密钥自动降级为 no-op，不影响主流程。
"""
from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from typing import Any, Callable

from ..config import settings

logger = logging.getLogger("agent.observability")

# 一次请求内的 request_id（协程安全；子协程自动继承）
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class RequestIdFilter(logging.Filter):
    """把当前 request_id 注入日志 extra['request_id']，便于按请求聚合检索。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


# ============================================================
#  request_id 工具
# ============================================================
def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_id(rid: str | None = None) -> str:
    """设置当前请求的 request_id；未传则生成新的。返回当前 request_id。"""
    rid = rid or new_request_id()
    _request_id_ctx.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id_ctx.get()


def log_event(event: str, **fields: Any) -> None:
    """结构化打点：`event` + 任意的 key-value，统一带上 request_id。"""
    logger.info(
        "EVENT request_id=%s event=%s fields=%s",
        get_request_id(),
        event,
        fields,
    )


# ============================================================
#  Langfuse 门面（可选，降级为 no-op）
# ============================================================
class LangfuseObserver:
    """Langfuse 追踪门面：未启用时方法全为 no-op（零开销降级）。"""

    def __init__(self) -> None:
        self._enabled: bool | None = None
        self._client: Any = None

    def is_enabled(self) -> bool:
        if self._enabled is None:
            self._enabled = self._try_enable()
        return self._enabled

    def _try_enable(self) -> bool:
        # 本版默认关闭；如需开启，在 .env 配置 LANGFUSE_* 并编写 driver。
        # 为降低依赖耦合，multi-agent 版直接用结构化日志（log_event），不再强绑 Langfuse。
        return False

    def observe(self, func: Callable) -> Callable:
        return func

    def update_trace(self, **kwargs: Any) -> None:  # no-op
        return

    def flush(self) -> None:  # no-op
        return


# 全局单例（保留 HELLO 同款命名，便于复用调用方）
observer = LangfuseObserver()


def setup_logging() -> None:
    """初始化日志：INFO 等级 + 注入 request_id Filter + 统一格式。"""
    # Windows 默认 stdout 是 GBK 编码，日志含 emoji/非 ASCII 会 UnicodeEncodeError。
    # 重配成 UTF-8（errors=replace 兜底），保证任何字符都能落日志（否则日志直接崩，掩盖真问题）。
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | rid=%(request_id)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("agent")
    root.setLevel(settings.log_level.upper())
    # 避免重复添加 handler（reload 场景）
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.handlers.clear()
        root.addHandler(handler)
