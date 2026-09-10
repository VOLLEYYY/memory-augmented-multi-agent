"""限流 —— slowapi 的 Limiter（按客户端 IP）。

独立成模块避免 `main.py ↔ routes.py` 的循环 import：limiter 在这里建一次，
`main.py` 挂 429 异常处理、`routes.py` 用 `@limiter.limit` 装饰。
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# 按客户端 IP 限流（get_remote_address 从 request.client 取 IP；走反代时要配 X-Forwarded-For）
limiter = Limiter(key_func=get_remote_address)
