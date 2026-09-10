"""Day10 真实验证：评审物理隔离（MODEL_B）+ 流式 review 事件。

用法：
  python verify_day10_http.py forward   # 默认：验证 MODEL_B 出分 + 流式含 review 事件
  python verify_day10_http.py reverse   # 验证 MODEL_B key 错 → 评审降级（证明评审确实走 B 而非 A）

reverse 模式配合「用环境变量覆盖 MODEL_B_API_KEY=wrong 起服务」使用，
用来证明 /ask 的评审链路走的是 MODEL_B 的 key（A 的 key 是对的，若偷偷走 A 就不会 401）。
"""
from __future__ import annotations

import json
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000"
Q = "我买的智能门锁A100保修期多久？"


def _post(question: str, thread_id: str, streaming: bool = False):
    data = json.dumps({"question": question, "thread_id": thread_id, "streaming": streaming}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/v1/ask", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=240)


def forward() -> int:
    fails = 0

    def check(cond, label):
        nonlocal fails
        print(("✅" if cond else "❌") + " " + label)
        if not cond:
            fails += 1

    # ---- 1. 非流式：MODEL_B 出分 ----
    r = _post(Q, "d10-fwd-1")
    try:
        d = json.loads(r.read().decode("utf-8"))
    finally:
        r.close()
    print("=" * 64)
    print(f"[1] 非流式 ask | review_score={d.get('review_score')}")
    print("    review_comment:", d.get("review_comment"))
    check(d.get("review_score") is not None, "MODEL_B 出分（review_score 非空）")
    check("维度评分" in d.get("review_comment", "") or "总分" in d.get("review_comment", ""),
          "review_comment 含评分维度")

    # ---- 2. 流式：含 review 事件 ----
    r2 = _post(Q, "d10-fwd-2", streaming=True)
    events = []
    try:
        for line in r2:
            line = line.decode("utf-8").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
    finally:
        r2.close()
    review_events = [e for e in events if e.get("type") == "review"]
    print("=" * 64)
    print(f"[2] 流式 ask | 事件总数={len(events)}，review 事件数={len(review_events)}")
    if review_events:
        print("    review 事件:", json.dumps(review_events[0], ensure_ascii=False)[:220])
    check(len(review_events) >= 1, "流式事件流含 review 事件")
    check(review_events and review_events[0].get("review_score") is not None,
          "review 事件含 review_score")

    print("=" * 64)
    print("结果：" + ("全部通过" if fails == 0 else f"{fails} 项未通过"))
    return 0 if fails == 0 else 1


def reverse() -> int:
    r = _post(Q, "d10-rev-1")
    try:
        d = json.loads(r.read().decode("utf-8"))
    finally:
        r.close()
    print("=" * 64)
    print(f"反向验证（MODEL_B key 错）| review_score={d.get('review_score')}")
    print("    review_comment:", d.get("review_comment"))
    print("    answer[:80]   :", (d.get("answer") or "")[:80])
    comment = d.get("review_comment", "")
    # 评审降级：score 空 + comment 提到「模型B」且（失败/未配置），answer 仍返回模型A 原始回答
    ok = (d.get("review_score") is None) and ("模型B" in comment) and bool(d.get("answer"))
    print("=" * 64)
    print(("✅" if ok else "❌") + " 评审确实走 MODEL_B（key 错 → 降级，但模型A 回答不受影响）")
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "forward"
    sys.exit(reverse() if mode == "reverse" else forward())
