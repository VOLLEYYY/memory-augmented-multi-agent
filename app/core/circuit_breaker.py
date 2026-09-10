"""熔断器 —— 三态（CLOSED/OPEN/HALF_OPEN）+ 冷却退避 + 半开试探。

对应 HELLO 复盘待优化点「无熔断器」：外部 LLM 服务抖动时，若不熔断，每个请求都会
卡到超时才失败，导致连接/线程被占满、雪崩。熔断器让系统在连续失败 N 次后**快速失败**
（直接抛错、不真调外部），冷却后再放一个「半开」试探请求探活。

状态机（面试点：三态 + 退避 + 半开只放一个试探）：
    CLOSED  --连续失败 >= threshold-->  OPEN
    OPEN    --冷却 recovery_timeout 秒--> HALF_OPEN（惰性转换，下次 allow_request 触发）
    HALF_OPEN --试探成功--> CLOSED（重置）
              --试探失败--> OPEN（重新计时冷却）

零依赖（只用 stdlib），可独立单测。
"""
from __future__ import annotations

import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"        # 正常放行
    OPEN = "open"            # 熔断，快速失败
    HALF_OPEN = "half_open"  # 冷却结束，放一个试探请求探活


class CircuitBreakerError(Exception):
    """熔断打开时抛出的异常——上层据此「快速失败」降级，而非真去调用外部服务。"""


class CircuitBreaker:
    """三态熔断器。

    Args:
        failure_threshold: 连续失败多少次后熔断（默认 5）。
        recovery_timeout: 熔断后的冷却秒数，冷却结束进入半开（默认 60s）。
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_trial = False  # 半开试探是否已放行（半开只放一个）

    @property
    def state(self) -> CircuitState:
        # OPEN 且过了冷却期 → 惰性转 HALF_OPEN（靠请求驱动，不引入后台线程）
        if self._state is CircuitState.OPEN and time.time() - self._opened_at >= self.recovery_timeout:
            self._state = CircuitState.HALF_OPEN
        return self._state

    def allow_request(self) -> bool:
        """是否放行本次调用。CLOSED 放行；OPEN 拒绝；HALF_OPEN 只放行一个试探。"""
        s = self.state
        if s is CircuitState.OPEN:
            return False
        if s is CircuitState.HALF_OPEN:
            if self._half_open_trial:
                return False  # 半开只放一个试探，其余仍拒绝
            self._half_open_trial = True
            return True
        return True  # CLOSED

    def record_success(self) -> None:
        """调用成功：重置失败计数、回到 CLOSED。"""
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._half_open_trial = False

    def record_failure(self) -> None:
        """调用失败：累加失败；半开试探失败则重新熔断；连续失败达阈值则熔断。"""
        if self._state is CircuitState.HALF_OPEN:
            # 半开试探失败 → 重新熔断、重新计时冷却
            self._state = CircuitState.OPEN
            self._opened_at = time.time()
            self._half_open_trial = False
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.time()
            self._half_open_trial = False
