# -*- coding: utf-8 -*-
"""
Circuit Breaker 熔断器

实现三态迁移模式:
  CLOSED (正常) → OPEN (熔断) → HALF_OPEN (试探恢复)

当下游服务（如 LLM 推理、外部搜索 API）连续失败达到阈值时，
自动切换到 OPEN 状态，拒绝新请求并返回降级响应，
避免雪崩式故障和无意义的重试开销。

经过冷却窗口后进入 HALF_OPEN 状态：
  - 放行一个试探请求
  - 成功 → 回到 CLOSED，恢复正常服务
  - 失败 → 回到 OPEN，重新计时

设计为装饰器 + 上下文管理器双模式，方便集成。
"""
import time
import threading
from enum import Enum
from typing import Optional, Callable, Any
from functools import wraps


class CircuitState(Enum):
    """熔断器状态枚举"""
    CLOSED = "CLOSED"          # 电路闭合 → 正常放行
    OPEN = "OPEN"              # 电路断开 → 拒绝请求
    HALF_OPEN = "HALF_OPEN"    # 半开 → 试探性放行单个请求


class CircuitBreakerOpenError(Exception):
    """熔断器处于 OPEN 状态时抛出的异常"""
    def __init__(self, breaker_name: str, retry_after: float):
        self.breaker_name = breaker_name
        self.retry_after = retry_after
        super().__init__(
            f"[CircuitBreaker:{breaker_name}] 熔断器已打开，"
            f"请在 {retry_after:.1f}s 后重试"
        )


class CircuitBreaker:
    """
    熔断器实现

    使用示例:
        # 方式 1: 装饰器
        breaker = CircuitBreaker("llm_service", failure_threshold=3)

        @breaker
        def call_llm(prompt):
            return llm.invoke(prompt)

        # 方式 2: 上下文管理器
        with breaker.guard():
            result = llm.invoke(prompt)

        # 方式 3: 手动调用
        breaker.before_call()
        try:
            result = llm.invoke(prompt)
            breaker.on_success()
        except Exception as e:
            breaker.on_failure()
            raise
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 1,
        fallback: Optional[Callable] = None,
    ):
        """
        Args:
            name: 熔断器名称（用于日志标识）
            failure_threshold: 连续失败多少次后触发 OPEN
            recovery_timeout: OPEN 状态持续多少秒后进入 HALF_OPEN
            success_threshold: HALF_OPEN 状态下连续成功多少次后回到 CLOSED
            fallback: 熔断时的降级函数，接收原始参数，返回降级结果
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.fallback = fallback

        # 状态
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

        # 统计指标（面试可讲）
        self.total_calls = 0
        self.total_failures = 0
        self.total_short_circuits = 0  # 被熔断拦截的请求数

    @property
    def state(self) -> CircuitState:
        """
        获取当前状态

        注意：OPEN 状态会自动检查是否已过冷却期，
        若已过期则自动迁移到 HALF_OPEN。
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    print(
                        f"🔄 [CircuitBreaker:{self.name}] "
                        f"冷却期结束({self.recovery_timeout}s)，进入 HALF_OPEN 试探状态"
                    )
            return self._state

    def before_call(self) -> None:
        """
        在调用目标服务之前调用

        Raises:
            CircuitBreakerOpenError: 熔断器处于 OPEN 状态时
        """
        current_state = self.state  # 触发 OPEN → HALF_OPEN 检查

        with self._lock:
            self.total_calls += 1

            if current_state == CircuitState.OPEN:
                self.total_short_circuits += 1
                retry_after = self.recovery_timeout - (
                    time.monotonic() - self._last_failure_time
                )
                raise CircuitBreakerOpenError(self.name, max(0, retry_after))

            # CLOSED 或 HALF_OPEN 放行

    def on_success(self) -> None:
        """调用成功时的回调"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    print(
                        f"✅ [CircuitBreaker:{self.name}] "
                        f"试探成功，恢复到 CLOSED 正常状态"
                    )
            else:
                # CLOSED 状态：重置失败计数
                self._failure_count = 0

    def on_failure(self) -> None:
        """调用失败时的回调"""
        with self._lock:
            self._failure_count += 1
            self.total_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # HALF_OPEN 下失败 → 直接回到 OPEN
                self._state = CircuitState.OPEN
                print(
                    f"🔴 [CircuitBreaker:{self.name}] "
                    f"HALF_OPEN 试探失败，回退到 OPEN（冷却 {self.recovery_timeout}s）"
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                print(
                    f"🔴 [CircuitBreaker:{self.name}] "
                    f"连续失败 {self._failure_count} 次，触发熔断 → OPEN"
                )

    def get_metrics(self) -> dict:
        """获取统计指标（可暴露到 /health 接口）"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "total_calls": self.total_calls,
                "total_failures": self.total_failures,
                "total_short_circuits": self.total_short_circuits,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
            }

    def __call__(self, func: Callable) -> Callable:
        """
        装饰器模式

        用法:
            @circuit_breaker
            def unstable_call():
                ...
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.before_call()
            try:
                result = func(*args, **kwargs)
                self.on_success()
                return result
            except CircuitBreakerOpenError:
                # 已经是打开的，尝试降级
                if self.fallback:
                    return self.fallback(*args, **kwargs)
                raise
            except Exception as e:
                self.on_failure()
                # 尝试降级
                if self.fallback and self._state == CircuitState.OPEN:
                    return self.fallback(*args, **kwargs)
                raise

        # 附加引用，方便外部获取状态
        wrapper.circuit_breaker = self
        return wrapper

    class _Guard:
        """上下文管理器内部类"""
        def __init__(self, breaker: 'CircuitBreaker'):
            self._breaker = breaker

        def __enter__(self):
            self._breaker.before_call()
            return self._breaker

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                self._breaker.on_success()
            elif exc_type is not CircuitBreakerOpenError:
                self._breaker.on_failure()
            return False  # 不吞异常

    def guard(self) -> '_Guard':
        """上下文管理器模式"""
        return self._Guard(self)
