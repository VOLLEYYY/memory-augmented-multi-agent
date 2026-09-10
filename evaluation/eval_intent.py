"""量化 4（售后专属）：意图分流准确率 + 自助解决率。

复用 graph._route 同款的启发式路由（tools.resolve_tool_call）：
  咨询 → qa（知识库解答）；订单 → query_order；退换 → apply_refund；物流 → query_logistics；投诉 → qa（升级）。

指标口径：
- 分流准确率 = 路由（qa/tools）与期望一致的问题占比；工具类进一步校验工具名是否命中。
- 自助解决率 = 无需人工升级即可自助解决（走工具或知识库）的问题占比；投诉类需转人工计为不可自助。
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.common import questions  # noqa: E402


def main() -> dict:
    from app.subagents.tools import resolve_tool_call

    qs = questions()
    total = len(qs)

    route_correct = 0
    tool_correct = 0
    tool_total = 0
    per_intent = defaultdict(lambda: {"total": 0, "correct": 0})
    self_service_count = 0

    rows = []
    for q in qs:
        tool_name, _args = resolve_tool_call(q["question"])
        pred_route = "tools" if tool_name else "qa"
        ok_route = pred_route == q["route_expected"]
        ok_tool = None
        if q["route_expected"] == "tools":
            tool_total += 1
            ok_tool = tool_name == q["tool_expected"]
            if ok_tool:
                tool_correct += 1
        if ok_route:
            route_correct += 1
        per_intent[q["intent"]]["total"] += 1
        if ok_route:
            per_intent[q["intent"]]["correct"] += 1
        if q["self_service"]:
            self_service_count += 1
        rows.append({
            "id": q["id"], "question": q["question"], "intent": q["intent"],
            "pred_route": pred_route, "pred_tool": tool_name,
            "route_ok": ok_route, "tool_ok": ok_tool,
        })

    route_acc = route_correct / total
    tool_acc = tool_correct / tool_total if tool_total else None
    self_service_rate = self_service_count / total

    print(f"评测集：{total} 条售后问题")
    print("=" * 70)
    print(f"意图分流准确率（route 维度）：{route_correct}/{total} = {route_acc:.1%}")
    if tool_acc is not None:
        print(f"工具名命中准确率（tools 维度）：{tool_correct}/{tool_total} = {tool_acc:.1%}")
    print(f"自助解决率（无需人工）：{self_service_count}/{total} = {self_service_rate:.1%}")
    print("-" * 70)
    print("分意图准确率：")
    for intent in ["咨询", "订单", "退换", "物流", "投诉"]:
        d = per_intent[intent]
        acc = d["correct"] / d["total"] if d["total"] else 0.0
        print(f"  {intent:<4} {d['correct']}/{d['total']} = {acc:.1%}")
    print("-" * 70)
    print("错误样例（如有）：")
    for r in rows:
        if not r["route_ok"] or r["tool_ok"] is False:
            print(f"  [{r['id']}] 「{r['question']}」 intent={r['intent']} "
                  f"预测={r['pred_tool'] or r['pred_route']}")

    return {
        "total": total,
        "route_accuracy": round(route_acc, 4),
        "tool_accuracy": round(tool_acc, 4) if tool_acc is not None else None,
        "self_service_rate": round(self_service_rate, 4),
        "per_intent": {k: {"total": v["total"], "correct": v["correct"]} for k, v in per_intent.items()},
    }


if __name__ == "__main__":
    main()
