"""用 Python 直接打 /ask 接口（避免 Windows curl GBK 编码问题）。"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"


def ask(question, thread_id="t"):
    data = json.dumps({"question": question, "thread_id": thread_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/v1/ask", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "什么是混合检索？"
    tid = sys.argv[2] if len(sys.argv) > 2 else "day4-t1"
    d = ask(q, tid)
    print("answer:", d.get("answer", "")[:400])
    print("sources:", d.get("sources"))
    print("review_score:", d.get("review_score"))
    print("need_revision:", d.get("need_revision"))
    print("route:", d.get("route"))
    print("memory_hit:", d.get("memory_hit"))
