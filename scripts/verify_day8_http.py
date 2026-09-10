"""Day8 真实验证脚本：通过 HTTP 打真实 uvicorn 服务，验证跨会话/跨重启/跨 thread 记忆。

配合「起服务 → 打第1轮 → 重启 → 打第2/3轮」的流程使用。
用法：python verify_day8_http.py "<question>" "<thread_id>"
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


if __name__ == "__main__":
    q, tid = sys.argv[1], sys.argv[2]
    d = ask(q, tid)
    print("=" * 64)
    print("question       :", q)
    print("thread_id      :", tid)
    print("memory_hit     :", d.get("memory_hit"))
    print("route          :", d.get("route"))
    print("sources        :", d.get("sources"))
    print("review_score   :", d.get("review_score"))
    print("memory_context :")
    print("  " + (d.get("memory_context") or "（空）").replace("\n", "\n  "))
    print("answer[:200]   :", (d.get("answer") or "")[:200])
