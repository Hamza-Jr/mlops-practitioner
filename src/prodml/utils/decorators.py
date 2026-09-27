import time
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

prediction_latency_ms: ContextVar[float | None] = ContextVar(
    "prediction_latency_ms",
    default=None,
)


def timed(func: F) -> F:
    """Measure function execution time in milliseconds."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            prediction_latency_ms.set(round(duration_ms, 2))

    return wrapper  # type: ignore[return-value]
