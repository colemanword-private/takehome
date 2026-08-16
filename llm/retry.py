from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _no_status_code(_: BaseException) -> None:
    return None


@dataclass(frozen=True)
class RetryEvent:
    """Describes a retry before it sleeps; attempt is the one-based retry number."""

    attempt: int
    delay_seconds: float
    status_code: int | None
    error_type: str


@dataclass(frozen=True)
class RetryPolicy:
    """Provider-independent limits for exponential backoff with full jitter."""

    max_retries: int
    base_delay_seconds: float
    max_delay_seconds: float

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if not math.isfinite(self.base_delay_seconds) or self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be finite and non-negative")
        if not math.isfinite(self.max_delay_seconds):
            raise ValueError("max_delay_seconds must be finite")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be at least base_delay_seconds")

    def delay_for(
        self,
        retry_index: int,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> float:
        """Return a full-jitter delay for a zero-based retry index."""
        if retry_index < 0:
            raise ValueError("retry_index must not be negative")
        if self.base_delay_seconds == 0:
            return 0.0
        try:
            exponential_delay = math.ldexp(self.base_delay_seconds, retry_index)
        except OverflowError:
            exponential_delay = self.max_delay_seconds
        maximum = min(self.max_delay_seconds, exponential_delay)
        return jitter(0.0, maximum)


async def retry_with_backoff(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    handled_errors: tuple[type[BaseException], ...],
    is_retryable: Callable[[BaseException], bool],
    status_code: Callable[[BaseException], int | None] = _no_status_code,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
    on_retry: Callable[[RetryEvent], None] | None = None,
) -> T:
    """Run an async operation while delegating error semantics to its provider."""
    for retry_index in range(policy.max_retries + 1):
        try:
            return await operation()
        except asyncio.CancelledError:
            # Cancellation belongs to the caller, even if BaseException is handled.
            raise
        except handled_errors as error:
            if retry_index == policy.max_retries or not is_retryable(error):
                raise

            event = RetryEvent(
                attempt=retry_index + 1,
                delay_seconds=policy.delay_for(retry_index, jitter),
                status_code=status_code(error),
                error_type=type(error).__name__,
            )
            _report_retry(on_retry, event)
            await sleep(event.delay_seconds)

    raise AssertionError("retry loop exhausted without returning or raising")


def _report_retry(
    observer: Callable[[RetryEvent], None] | None,
    event: RetryEvent,
) -> None:
    if observer is None:
        return
    try:
        observer(event)
    except Exception:
        # Telemetry must never turn a recoverable provider failure into an outage.
        logger.exception("retry observer failed")
