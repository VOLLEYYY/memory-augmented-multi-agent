"""工具执行子 Agent —— 白名单调度 + 售后业务工具 + 安全闸门。

职责（对应 HELLO「无 Function Calling」待优化点的落地）：
1. `run_tool(services, name, args)` —— 按目录调度工具，且只允许执行 .env 白名单内的工具，
   防注入（面试可讲"工具白名单 + 参数校验"，对应阿里"删库"题）。
2. 售后业务工具（Day 9 落地，本地 mock，模拟真实订单/物流/退款系统 API）：
   - `query_order(order_id)`      查订单状态（待发货 / 运输中 / 已签收）
   - `query_logistics(tracking_no)` 查物流轨迹
   - `apply_refund(order_id, reason)` 发起退款（返回退款单号）
   本地用 mock 隔离外部依赖，换真实 key/endpoint 即切换（面试讲"为什么 mock"）。
3. `resolve_tool_call(question)` —— 启发式意图路由（关键词 → 工具名 + 参数）。
4. `to_langchain_tools()` + `run_with_function_calling()` —— 加分项：用 `bind_tools`
   让模型自主决定调哪个工具（tool_call → 执行 → 回填 → 续生成），失败退回启发式路由。
5. `safety_gate(services, text)` —— 对生成结果做规则安检，命中则拦截。
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..config import settings

logger = logging.getLogger("agent.subagents.tools")


# ============================================================
#  售后 mock 工具（模拟真实订单/物流/退款系统 API）
# ============================================================
# 用「字符码求和」而非内置 hash() 做确定性选择 —— hash() 受 PYTHONHASHSEED 影响跨进程随机，
# 会导致同一订单号每次结果不一样（延续坑 8 的教训）。
def _stable_index(key: str, n: int) -> int:
    return sum(ord(c) for c in key) % n


async def _query_order(order_id: str) -> Dict[str, Any]:
    """查订单状态（mock）。返回确定性的假数据，模拟订单系统 API。"""
    statuses = ["待发货", "运输中", "已签收"]
    status = statuses[_stable_index(order_id, len(statuses))]
    return {
        "tool": "query_order",
        "order_id": order_id,
        "status": status,
        "items": [{"sku": "HS-SmartLock-A100", "name": "星屋智能门锁 A100", "qty": 1}],
        "note": "mock 数据：模拟订单系统 API，接入真实系统时替换",
    }


async def _query_logistics(tracking_no: str) -> Dict[str, Any]:
    """查物流轨迹（mock）。模拟物流系统 API。"""
    return {
        "tool": "query_logistics",
        "tracking_no": tracking_no,
        "status": "运输中",
        "traces": [
            {"time": "2026-09-05 09:12", "desc": "包裹已揽收，发往分拨中心"},
            {"time": "2026-09-06 03:40", "desc": "到达【华东分拨中心】"},
            {"time": "2026-09-07 08:05", "desc": "运输中，预计 2 日内送达"},
        ],
        "note": "mock 数据：模拟物流系统 API",
    }


async def _apply_refund(order_id: str, reason: str = "用户申请退款") -> Dict[str, Any]:
    """发起退款（mock）。返回退款单号，模拟售后/支付系统 API。"""
    refund_no = f"RF{int(time.time() * 1000)}"
    return {
        "tool": "apply_refund",
        "order_id": order_id,
        "reason": reason,
        "refund_no": refund_no,
        "status": "已受理，1-3 个工作日原路退回",
        "note": "mock 数据：模拟售后/支付系统 API",
    }


# ---- 占位工具（可选加分：TAVILY_API_KEY 已配，可换真实联网搜索）----
async def _web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    return {
        "tool": "web_search",
        "results": [{"title": f"关于「{query}」的结果{t}", "url": "https://example.com"} for t in range(max_results)],
        "note": "占位实现：配置 TAVILY_API_KEY 后接入真实搜索",
    }


# 已实现但**不在白名单内**的工具 —— 用于演示「注册了但未授权 → 拒绝执行」。
async def _weather(city: str) -> Dict[str, Any]:
    return {"tool": "weather", "city": city, "temperature": "未知（占位）"}


_TOOL_REGISTRY = {
    "query_order": _query_order,
    "query_logistics": _query_logistics,
    "apply_refund": _apply_refund,
    "web_search": _web_search,
    "weather": _weather,          # 注册但不在白名单（演示白名单拦截）
    "safety_check": None,         # safety 由 safety_gate 处理，不在这里调度
}


# ============================================================
#  意图路由：售后问题 → 工具名 + 参数（启发式；加分项用 Function Calling 替代）
# ============================================================
_REFUND_KEYWORDS = ("退款", "退货", "退换", "退钱", "申请退款")
_LOGISTICS_KEYWORDS = ("物流", "快递", "包裹", "到哪了", "查物流", "查快递")
_ORDER_KEYWORDS = ("订单", "查订单", "下单", "订单状态")

_ORDER_ID_RE = re.compile(r"([A-Za-z]{1,4}\d{3,})")
_TRACKING_RE = re.compile(r"([A-Z]{2,3}\d{8,})", re.IGNORECASE)


def _extract_reason(q: str) -> str:
    m = re.search(r"(?:因为|由于|原因[是为:]?)(.{0,20})", q)
    return (m.group(1).strip() or "用户申请退款") if m else "用户申请退款"


def resolve_tool_call(question: str) -> tuple[Optional[str], Dict[str, Any]]:
    """启发式路由：问题 → (工具名, 参数)。未命中返回 (None, {})。

    匹配顺序有讲究：退款优先于物流/订单（"对订单 A123 申请退款"同时含"订单"和"退款"，
    应走退款）；物流优先于订单（"查订单的物流"含"物流"）。
    """
    if any(kw in question for kw in _REFUND_KEYWORDS):
        m = _ORDER_ID_RE.search(question)
        return "apply_refund", {"order_id": m.group(1) if m else "A000",
                                "reason": _extract_reason(question)}
    if any(kw in question for kw in _LOGISTICS_KEYWORDS):
        m = _TRACKING_RE.search(question)
        return "query_logistics", {"tracking_no": m.group(1) if m else "SF0000000000"}
    if any(kw in question for kw in _ORDER_KEYWORDS):
        m = _ORDER_ID_RE.search(question)
        return "query_order", {"order_id": m.group(1) if m else "A000"}
    return None, {}


# ============================================================
#  投诉/转人工 路由（Day 13：评测暴露「投诉类未转人工」缺口后补上）
# ============================================================
_ESCALATE_KEYWORDS = ("投诉", "差评", "索赔", "转人工", "找人工", "人工客服", "升级处理", "态度差")


def resolve_escalation(question: str) -> bool:
    """判断是否需要转人工（投诉 / 差评 / 索赔 / 升级 / 找人工）。

    Day 12 评测（量化 4）暴露的缺口：投诉类问题此前被当成普通咨询自助回答，
    缺「投诉→转人工」路由。这里补上：含投诉诉求即转人工，优先级最高（在工具/QA 之前判）。
    """
    return any(kw in question for kw in _ESCALATE_KEYWORDS)


def format_escalation() -> str:
    """转人工话术（依据 `售后-售后服务流程.md`「投诉与升级」：升级后客服主管 24 小时内回电）。"""
    return (
        "已为您登记并升级处理：您的问题涉及投诉，已转接人工客服。"
        "客服主管将在 24 小时内回电跟进；如需更快联系，可拨打 400-800-1234（服务时间 9:00-18:00）。"
    )


# ============================================================
#  调度 + 结果格式化
# ============================================================
async def run_tool(services, name: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """调度工具。未注册 → 报错；注册但不在白名单 → 拒绝；参数错误 → 返回错误。"""
    args = args or {}
    if name not in _TOOL_REGISTRY:
        return {"tool": name, "error": f"未知工具 {name}（未注册）"}
    whitelist = settings.get_tool_whitelist()
    if name not in whitelist:
        return {"tool": name, "error": f"工具 {name} 不在白名单内，拒绝执行"}
    fn = _TOOL_REGISTRY[name]
    if fn is None:
        return {"tool": name, "error": "空实现"}
    try:
        return await fn(**args)
    except TypeError as e:
        return {"tool": name, "error": f"参数错误：{e}"}


def format_tool_result(result: Dict[str, Any]) -> str:
    """把工具返回的 dict 格式化为给用户的自然语言回答。"""
    if "error" in result:
        return f"[工具] {result['error']}"
    name = result.get("tool", "")
    if name == "query_order":
        items = "、".join(i.get("name", "") for i in result.get("items", [])) or "商品"
        return f"订单 {result['order_id']}（{items}）当前状态：{result['status']}。"
    if name == "query_logistics":
        traces = "\n".join(f"· {t['time']}  {t['desc']}" for t in result.get("traces", []))
        return f"物流单号 {result['tracking_no']} 当前状态：{result['status']}\n{traces}"
    if name == "apply_refund":
        return (f"退款申请已受理，退款单号：{result['refund_no']}"
                f"（订单 {result['order_id']}，原因：{result['reason']}）。")
    return str(result)


# ============================================================
#  加分项：bind_tools Function Calling（模型自主决定调哪个工具）
# ============================================================
def to_langchain_tools(whitelist: Optional[List[str]] = None) -> List[Any]:
    """把售后工具包装成 LangChain @tool（供 `model.bind_tools(...)` 声明工具 schema）。

    工具体里用 asyncio.run 兜底直调 —— 实际 Function Calling 流程由
    `run_with_function_calling` 自己执行 `run_tool`，这里只取签名与 docstring 作 schema。
    """
    import asyncio

    from langchain_core.tools import tool

    def _run(coro, *a, **kw):
        return asyncio.run(coro(*a, **kw))

    @tool
    def query_order_tool(order_id: str) -> str:
        """查询订单状态。order_id：订单号，例如 A123。"""
        return format_tool_result(_run(_query_order, order_id))

    @tool
    def query_logistics_tool(tracking_no: str) -> str:
        """查询物流轨迹。tracking_no：物流单号。"""
        return format_tool_result(_run(_query_logistics, tracking_no))

    @tool
    def apply_refund_tool(order_id: str, reason: str) -> str:
        """发起退款申请。order_id：订单号；reason：退款原因。"""
        return format_tool_result(_run(_apply_refund, order_id, reason))

    return [query_order_tool, query_logistics_tool, apply_refund_tool]


async def run_with_function_calling(services, question: str) -> Optional[str]:
    """加分项：bind_tools Function Calling。模型自主决定调哪个工具 → 执行 → 回填 → 续生成。

    流程（面试讲点）：tool_call 生成 → 白名单执行 → ToolMessage 回填 → 二跳续生成。
    任何环节失败（无模型/不支持工具调用）返回 None，由调用方退回启发式路由。
    """
    try:
        from ..core.llm import get_chat_model
        from langchain_core.messages import HumanMessage, ToolMessage
    except ImportError:
        return None
    try:
        model = get_chat_model("a").bind_tools(to_langchain_tools())
        resp = await model.ainvoke([HumanMessage(content=question)])
    except Exception as e:
        logger.warning("Function Calling 首跳失败（%s），退回启发式路由", e.__class__.__name__)
        return None

    tool_calls = getattr(resp, "tool_calls", None) or []
    if not tool_calls:
        return None  # 模型未选择工具，退回启发式路由

    tool_msgs = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}
        result = await run_tool(services, name, args)
        tool_msgs.append(
            ToolMessage(
                content=format_tool_result(result),
                tool_call_id=tc.get("id", ""),
                name=name,
            )
        )
    try:
        final = await model.ainvoke([HumanMessage(content=question), resp, *tool_msgs])
        return str(final.content or "")
    except Exception as e:
        logger.warning("Function Calling 二跳失败（%s），退回启发式路由", e.__class__.__name__)
        return None


async def safety_gate(services, text: str) -> Dict[str, Any]:
    """对生成结果做规则安检（复用 HELLO SafetyTool）。"""
    return await services.safety.execute(text)
