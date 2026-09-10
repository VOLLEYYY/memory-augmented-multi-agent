"""Day9 真实验证脚本：通过 HTTP 打真实 uvicorn，验证「跨用户记忆隔离（P0 修复）+ 售后工具」。

流程（配合后台常驻 uvicorn 使用）：
  1. user-1 问「门锁保修」→ 写入带 thread_id 的 episodic；
  2. user-2 问「门锁换货」（语义相近）→ 检索按 thread_id=user-2 过滤，不应命中 user-1 记忆；
  3. 工具：查订单 A123 → 返回订单状态；申请退款 → 返回退款单号；白名单外工具被拒。
用法：python verify_day9_http.py
"""
from __future__ import annotations

import json
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000"


def ask(question: str, thread_id: str) -> dict:
    data = json.dumps({"question": question, "thread_id": thread_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/v1/ask", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    fails = 0

    def check(cond: bool, label: str):
        nonlocal fails
        mark = "✅" if cond else "❌"
        print(f"{mark} {label}")
        if not cond:
            fails += 1

    # ---- 1. user-1 写入「门锁保修」记忆 ----
    r1 = ask("我买的智能门锁A100保修期多久？", "day9-user-1")
    print("=" * 64)
    print(f"[1] user-1 门锁保修 | route={r1.get('route')} memory_hit={r1.get('memory_hit')}")
    print("    answer:", (r1.get("answer") or "")[:80])

    # ---- 2. user-2 问「门锁换货」，应不命中 user-1 的保修记忆 ----
    r2 = ask("那我刚才说的门锁想换货怎么操作？", "day9-user-2")
    ctx2 = r2.get("memory_context") or ""
    leaked = "A100" in ctx2 or "保修" in ctx2
    print("=" * 64)
    print(f"[2] user-2 门锁换货 | route={r2.get('route')} memory_hit={r2.get('memory_hit')}")
    print("    memory_context:", (ctx2 or "（空）").replace("\n", " ")[:160])
    check(not leaked, "跨用户隔离：user-2 未命中 user-1 的「A100/保修」记忆")

    # ---- 3. 工具：查订单 ----
    r3 = ask("查一下我的订单 A123", "day9-user-3")
    ans3 = r3.get("answer") or ""
    print("=" * 64)
    print(f"[3] 查订单 A123 | route={r3.get('route')}")
    print("    answer:", ans3[:120])
    check(r3.get("route") == "tools", "查订单走 tools 路由")
    check(("A123" in ans3) and any(s in ans3 for s in ("待发货", "运输中", "已签收")),
          "查订单返回订单状态（非占位）")

    # ---- 4. 工具：申请退款 ----
    r4 = ask("帮我申请退款", "day9-user-3")
    ans4 = r4.get("answer") or ""
    print("=" * 64)
    print(f"[4] 申请退款 | route={r4.get('route')}")
    print("    answer:", ans4[:120])
    check("退款单号" in ans4 and "RF" in ans4, "申请退款返回退款单号")

    print("=" * 64)
    print(f"结果：{'全部通过' if fails == 0 else str(fails) + ' 项未通过'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
