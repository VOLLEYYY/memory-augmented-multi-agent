"""Day11 限流复验：连打 /ask，超过阈值后返回 429。

配合「RATE_LIMIT_ASK=3/minute 环境变量覆盖起服务」使用（默认 30/minute 不好触发）。
用工具类问题（查订单，走 mock 工具、不调 LLM）让请求很快，便于快速连打触发限流。
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000"


def ask(q: str, tid: str) -> int:
    data = json.dumps({"question": q, "thread_id": tid}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/v1/ask", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        r = urllib.request.urlopen(req, timeout=60)
        code = r.status
        r.close()
        return code
    except urllib.error.HTTPError as e:
        return e.code


def main() -> int:
    codes = []
    for i in range(1, 6):
        c = ask("查一下我的订单 A123", f"rl-{i}")
        codes.append(c)
        print(f"第{i}次: HTTP {c}")
    got_429 = any(c == 429 for c in codes)
    ok = got_429 and codes[0] == 200  # 首次放行，后续触发 429
    print("=" * 64)
    print(("✅" if ok else "❌") + " 限流生效（连打超过阈值后返回 429）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
