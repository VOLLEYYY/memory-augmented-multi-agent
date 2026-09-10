"""熔断器单测 —— 覆盖三态转换、熔断触发、半开试探、冷却恢复（Day 11）。

零依赖：只测 `CircuitBreaker` 纯状态机，不碰 LLM / qdrant / langgraph。
运行：cd multi-agent && pytest -q tests/test_circuit_breaker.py
"""
from __future__ import annotations

import time

from app.core.circuit_breaker import CircuitBreaker, CircuitState


def test_closed_allows_and_counts_failures():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
    assert cb.state is CircuitState.CLOSED
    # 连续失败 2 次还没到阈值，仍是 CLOSED、放行
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.CLOSED
    assert cb.allow_request() is True


def test_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.state is CircuitState.OPEN
    assert cb.allow_request() is False  # 熔断 → 快速失败


def test_success_resets_failure_count():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
    for _ in range(2):
        cb.record_failure()
    cb.record_success()
    assert cb.state is CircuitState.CLOSED
    # 失败计数已清零：再失败 2 次（共 2 次，不是 4 次）仍不熔断
    for _ in range(2):
        cb.record_failure()
    assert cb.state is CircuitState.CLOSED


def test_half_open_after_recovery_timeout():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    time.sleep(0.03)  # 等冷却期过
    assert cb.state is CircuitState.HALF_OPEN  # 惰性转半开
    assert cb.allow_request() is True  # 放一个试探


def test_half_open_success_closes():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    time.sleep(0.03)
    assert cb.allow_request() is True  # 试探放行
    cb.record_success()
    assert cb.state is CircuitState.CLOSED  # 试探成功 → 关闭


def test_half_open_failure_reopens():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    time.sleep(0.03)
    assert cb.allow_request() is True  # 试探
    cb.record_failure()  # 试探失败
    assert cb.state is CircuitState.OPEN  # 重新熔断


def test_half_open_only_one_trial():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    time.sleep(0.03)
    assert cb.allow_request() is True   # 第一个试探放行
    assert cb.allow_request() is False  # 第二个被拒（半开只放一个试探）
