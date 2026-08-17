from __future__ import annotations

import asyncio
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, Literal, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _no_status_code(_: BaseException) -> None:
    return None


def _no_retry_after(_: BaseException) -> None:
    return None


RetryExhaustionReason = Literal[
    "deadline",
    "non_retryable",
    "retry_budget",
    "retry_limit",
]


@dataclass(frozen=True)
class RetryEvent:
    """Describes an admitted retry; attempt is the one-based retry number."""

    attempt: int
    delay_seconds: float
    status_code: int | None
    error_type: str
    retry_after_seconds: float | None = None
    cumulative_backoff_seconds: float = 0.0


@dataclass(frozen=True)
class BackoffEvent:
    """Describes a completed retry backoff, whether or not retry was admitted."""

    delay_seconds: float
    status_code: int | None
    error_type: str
    retry_after_seconds: float | None = None
    cumulative_backoff_seconds: float = 0.0


@dataclass(frozen=True)
class RetryExhaustedEvent:
    """Describes why a request stopped retrying after one or more attempts."""

    attempts: int
    retries: int
    cumulative_backoff_seconds: float
    status_code: int | None
    error_type: str
    reason: RetryExhaustionReason
    elapsed_seconds: float


class RetryDeadlineExceeded(TimeoutError):
    """Raised when a request exhausts its end-to-end retry deadline."""

    def __init__(self, event: RetryExhaustedEvent) -> None:
        self.event = event
        self.status_code = event.status_code
        self.attempts = event.attempts
        self.cumulative_backoff_seconds = event.cumulative_backoff_seconds
        self.exhaustion_reason = event.reason
        super().__init__(
            f"request cannot continue within its deadline after "
            f"{event.attempts} attempts and {event.elapsed_seconds:.3f} seconds"
        )


class RetryBudget:
    """Process-local token bucket shared by requests through one provider instance."""

    def __init__(
        self,
        capacity: int,
        refill_rate_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if (
            not math.isfinite(refill_rate_per_second)
            or refill_rate_per_second < 0
        ):
            raise ValueError(
                "refill_rate_per_second must be finite and non-negative"
            )

        self._capacity = capacity
        self._refill_rate_per_second = refill_rate_per_second
        self._clock = clock
        self._tokens = float(capacity)
        self._last_refill = clock()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Consume one retry token if the budget currently has capacity."""
        with self._lock:
            self._refill()
            if self._tokens < 1.0:
                return False
            self._tokens -= 1.0
            return True

    @property
    def available_tokens(self) -> float:
        """Return currently available tokens, refilling first for observability."""
        with self._lock:
            self._refill()
            return self._tokens

    def _refill(self) -> None:
        now = self._clock()
        elapsed_seconds = max(0.0, now - self._last_refill)
        self._tokens = min(
            float(self._capacity),
            self._tokens + elapsed_seconds * self._refill_rate_per_second,
        )
        self._last_refill = now


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
    retry_after: Callable[[BaseException], float | None] = _no_retry_after,
    retry_budget: RetryBudget | None = None,
    total_timeout_seconds: float | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
    on_backoff: Callable[[BackoffEvent], None] | None = None,
    on_retry: Callable[[RetryEvent], None] | None = None,
    on_exhausted: Callable[[RetryExhaustedEvent], None] | None = None,
) -> T:
    """Run an async operation while delegating error semantics to its provider."""
    if total_timeout_seconds is not None and (
        not math.isfinite(total_timeout_seconds) or total_timeout_seconds <= 0
    ):
        raise ValueError("total_timeout_seconds must be finite and positive")

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    deadline_at = (
        started_at + total_timeout_seconds
        if total_timeout_seconds is not None
        else None
    )
    cumulative_backoff_seconds = 0.0

    def report_exhaustion(
        error: BaseException,
        reason: RetryExhaustionReason,
        attempts: int,
    ) -> RetryExhaustedEvent:
        event = _exhausted_event(
            attempts=attempts,
            cumulative_backoff_seconds=cumulative_backoff_seconds,
            error=error,
            reason=reason,
            status_code=status_code,
            elapsed_seconds=loop.time() - started_at,
        )
        _report_observer(on_exhausted, event, "retry exhaustion")
        return event

    for retry_index in range(policy.max_retries + 1):
        remaining_seconds = _remaining_seconds(deadline_at, loop.time())
        if remaining_seconds is not None and remaining_seconds <= 0:
            raise RetryDeadlineExceeded(
                report_exhaustion(TimeoutError(), "deadline", retry_index)
            )

        try:
            return await _await_with_timeout(operation(), remaining_seconds)
        except asyncio.CancelledError:
            # Cancellation belongs to the caller, even if BaseException is handled.
            raise
        except BaseException as error:
            deadline_expired = deadline_at is not None and loop.time() >= deadline_at
            if isinstance(error, TimeoutError) and deadline_expired:
                event = report_exhaustion(error, "deadline", retry_index + 1)
                raise RetryDeadlineExceeded(event) from error

            if not isinstance(error, handled_errors):
                raise

            if deadline_expired:
                event = report_exhaustion(error, "deadline", retry_index + 1)
                raise RetryDeadlineExceeded(event) from error

            if not is_retryable(error):
                report_exhaustion(error, "non_retryable", retry_index + 1)
                raise

            if retry_index == policy.max_retries:
                report_exhaustion(error, "retry_limit", retry_index + 1)
                raise

            provider_delay = _safe_retry_after(retry_after, error)
            policy_delay = policy.delay_for(retry_index, jitter)
            delay_seconds = max(policy_delay, provider_delay or 0.0)
            remaining_seconds = _remaining_seconds(deadline_at, loop.time())
            if (
                remaining_seconds is not None
                and delay_seconds >= remaining_seconds
            ):
                event = report_exhaustion(error, "deadline", retry_index + 1)
                raise RetryDeadlineExceeded(event) from error

            try:
                await _await_with_timeout(sleep(delay_seconds), remaining_seconds)
            except asyncio.CancelledError:
                raise
            except TimeoutError as sleep_error:
                if deadline_at is None or loop.time() < deadline_at:
                    raise
                raise RetryDeadlineExceeded(
                    report_exhaustion(error, "deadline", retry_index + 1)
                ) from sleep_error
            cumulative_backoff_seconds += delay_seconds
            _report_observer(
                on_backoff,
                BackoffEvent(
                    delay_seconds=delay_seconds,
                    status_code=status_code(error),
                    error_type=type(error).__name__,
                    retry_after_seconds=provider_delay,
                    cumulative_backoff_seconds=cumulative_backoff_seconds,
                ),
                "retry backoff",
            )

            remaining_seconds = _remaining_seconds(deadline_at, loop.time())
            if remaining_seconds is not None and remaining_seconds <= 0:
                event = report_exhaustion(error, "deadline", retry_index + 1)
                raise RetryDeadlineExceeded(event) from error

            if retry_budget is not None and not retry_budget.try_acquire():
                report_exhaustion(error, "retry_budget", retry_index + 1)
                raise

            event = RetryEvent(
                attempt=retry_index + 1,
                delay_seconds=delay_seconds,
                status_code=status_code(error),
                error_type=type(error).__name__,
                retry_after_seconds=provider_delay,
                cumulative_backoff_seconds=cumulative_backoff_seconds,
            )
            _report_observer(on_retry, event, "retry")

    raise AssertionError("retry loop exhausted without returning or raising")


def retry_after_seconds(error: BaseException) -> float | None:
    """Parse Retry-After or retry-after-ms from a provider HTTP response."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    retry_after = _header_value(headers, "retry-after")
    if retry_after is not None:
        try:
            delay_seconds = float(retry_after)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                delay_seconds = retry_at.timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                return None
        if math.isfinite(delay_seconds):
            return max(0.0, delay_seconds)
        return None

    retry_after_ms = _header_value(headers, "retry-after-ms")
    if retry_after_ms is None:
        return None
    try:
        delay_seconds = float(retry_after_ms) / 1_000
    except (TypeError, ValueError):
        return None
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        return None
    return delay_seconds


def _remaining_seconds(deadline_at: float | None, now: float) -> float | None:
    return None if deadline_at is None else deadline_at - now


async def _await_with_timeout(
    awaitable: Awaitable[T], timeout_seconds: float | None
) -> T:
    if timeout_seconds is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=timeout_seconds)


def _exhausted_event(
    *,
    attempts: int,
    cumulative_backoff_seconds: float,
    error: BaseException,
    reason: RetryExhaustionReason,
    status_code: Callable[[BaseException], int | None],
    elapsed_seconds: float,
) -> RetryExhaustedEvent:
    return RetryExhaustedEvent(
        attempts=attempts,
        retries=max(0, attempts - 1),
        cumulative_backoff_seconds=cumulative_backoff_seconds,
        status_code=status_code(error),
        error_type=type(error).__name__,
        reason=reason,
        elapsed_seconds=elapsed_seconds,
    )


def _safe_retry_after(
    retry_after: Callable[[BaseException], float | None],
    error: BaseException,
) -> float | None:
    try:
        delay_seconds = retry_after(error)
    except Exception:
        logger.exception("retry-after parser failed")
        return None
    if delay_seconds is None:
        return None
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        logger.warning("retry-after parser returned an invalid delay")
        return None
    return delay_seconds


def _header_value(headers: object, name: str) -> object | None:
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is not None:
            return value
    items = getattr(headers, "items", None)
    if not callable(items):
        return None
    for key, value in items():
        if str(key).casefold() == name.casefold():
            return value
    return None


EventT = TypeVar("EventT")


def _report_observer(
    observer: Callable[[EventT], None] | None,
    event: EventT,
    description: str,
) -> None:
    if observer is None:
        return
    try:
        observer(event)
    except Exception:
        # Telemetry must never turn a recoverable provider failure into an outage.
        logger.exception("%s observer failed", description)
