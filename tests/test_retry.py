from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from llm import (
    BackoffEvent,
    RetryBudget,
    RetryDeadlineExceeded,
    RetryEvent,
    RetryExhaustedEvent,
    RetryPolicy,
    retry_after_seconds,
    retry_with_backoff,
    status_code_from_error,
)

pytestmark = pytest.mark.asyncio


class ProviderError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"provider error {status}")
        self.status = status


async def test_status_code_from_error_prefers_numeric_code_over_status_text() -> None:
    # google-genai errors carry an int `code` plus a string `status` such as
    # "RESOURCE_EXHAUSTED"; the text must not shadow the number.
    error = RuntimeError("rate limited")
    error.code = 429  # type: ignore[attr-defined]
    error.status = "RESOURCE_EXHAUSTED"  # type: ignore[attr-defined]

    assert status_code_from_error(error) == 429


async def test_status_code_from_error_skips_unparseable_values() -> None:
    error = RuntimeError("unavailable")
    error.status = "UNAVAILABLE"  # type: ignore[attr-defined]
    error.status_code = 503  # type: ignore[attr-defined]

    assert status_code_from_error(error) == 503


async def test_status_code_from_error_accepts_numeric_status_strings() -> None:
    assert status_code_from_error(ProviderError(429)) == 429


async def test_status_code_from_error_returns_none_without_a_status() -> None:
    assert status_code_from_error(RuntimeError("boom")) is None


async def test_retries_with_full_jitter_and_reports_event() -> None:
    outcomes: list[str | BaseException] = [ProviderError(429), "ok"]
    delays: list[float] = []
    backoffs: list[BackoffEvent] = []
    events: list[RetryEvent] = []

    async def operation() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def sleep(delay: float) -> None:
        delays.append(delay)

    result = await retry_with_backoff(
        operation,
        policy=RetryPolicy(2, 0.5, 2.0),
        handled_errors=(ProviderError,),
        is_retryable=lambda error: error.status == 429,
        status_code=lambda error: error.status,
        sleep=sleep,
        jitter=lambda low, high: (low + high) / 2,
        on_backoff=backoffs.append,
        on_retry=events.append,
    )

    assert result == "ok"
    assert delays == [0.25]
    assert backoffs == [
        BackoffEvent(
            delay_seconds=0.25,
            status_code=429,
            error_type="ProviderError",
            cumulative_backoff_seconds=0.25,
        )
    ]
    assert events == [
        RetryEvent(
            attempt=1,
            delay_seconds=0.25,
            status_code=429,
            error_type="ProviderError",
            cumulative_backoff_seconds=0.25,
        )
    ]


async def test_provider_retry_after_takes_precedence_over_policy_delay() -> None:
    outcomes: list[str | BaseException] = [ProviderError(429), "ok"]
    delays: list[float] = []
    events: list[RetryEvent] = []

    async def operation() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    result = await retry_with_backoff(
        operation,
        policy=RetryPolicy(1, 0.5, 2.0),
        handled_errors=(ProviderError,),
        is_retryable=lambda _: True,
        retry_after=lambda _: 1.5,
        sleep=lambda delay: _record_delay(delays, delay),
        jitter=lambda _low, high: high,
        on_retry=events.append,
    )

    assert result == "ok"
    assert delays == [1.5]
    assert events[0].retry_after_seconds == 1.5
    assert events[0].cumulative_backoff_seconds == 1.5


async def test_exponential_schedule_reaches_and_holds_cap() -> None:
    outcomes: list[str | BaseException] = [
        ProviderError(503),
        ProviderError(503),
        ProviderError(503),
        ProviderError(503),
        ProviderError(503),
        "ok",
    ]
    delays: list[float] = []

    async def operation() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    result = await retry_with_backoff(
        operation,
        policy=RetryPolicy(5, 0.5, 2.0),
        handled_errors=(ProviderError,),
        is_retryable=lambda _: True,
        sleep=lambda delay: _record_delay(delays, delay),
        jitter=lambda _low, high: high,
    )

    assert result == "ok"
    assert delays == [0.5, 1.0, 2.0, 2.0, 2.0]


@pytest.mark.parametrize("max_retries", [0, 2])
async def test_stops_after_initial_attempt_plus_retry_budget(
    max_retries: int,
) -> None:
    calls = 0
    delays: list[float] = []

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise ProviderError(503)

    with pytest.raises(ProviderError):
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(max_retries, 0.5, 2.0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            sleep=lambda delay: _record_delay(delays, delay),
        )

    assert calls == max_retries + 1
    assert len(delays) == max_retries


async def test_does_not_retry_error_rejected_by_provider() -> None:
    error = ProviderError(400)

    async def operation() -> None:
        raise error

    with pytest.raises(ProviderError) as raised:
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(2, 0.5, 2.0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: False,
            status_code=lambda provider_error: provider_error.status,
        )

    assert raised.value is error


async def test_never_retries_cancellation() -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(2, 0.5, 2.0),
            handled_errors=(BaseException,),
            is_retryable=lambda _: True,
            status_code=lambda _: None,
        )

    assert calls == 1


async def test_cancellation_during_backoff_stops_retrying() -> None:
    calls = 0
    sleep_started = asyncio.Event()
    events: list[RetryEvent] = []
    budget = RetryBudget(capacity=1, refill_rate_per_second=0)

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise ProviderError(503)

    async def blocked_sleep(_: float) -> None:
        sleep_started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        retry_with_backoff(
            operation,
            policy=RetryPolicy(2, 0.5, 2.0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            retry_budget=budget,
            sleep=blocked_sleep,
            on_retry=events.append,
        )
    )
    await sleep_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == 1
    assert events == []
    assert budget.available_tokens == 1


async def test_observer_failure_does_not_interrupt_retry() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError(503)
        return "ok"

    def broken_observer(_: RetryEvent) -> None:
        raise RuntimeError("metrics unavailable")

    result = await retry_with_backoff(
        operation,
        policy=RetryPolicy(1, 0.0, 0.0),
        handled_errors=(ProviderError,),
        is_retryable=lambda _: True,
        status_code=lambda error: error.status,
        sleep=lambda _: asyncio.sleep(0),
        on_retry=broken_observer,
    )

    assert result == "ok"
    assert calls == 2


async def test_exhaustion_observer_failure_preserves_provider_error() -> None:
    error = ProviderError(503)

    async def operation() -> None:
        raise error

    def broken_observer(_: RetryExhaustedEvent) -> None:
        raise RuntimeError("metrics unavailable")

    with pytest.raises(ProviderError) as raised:
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(0, 0.0, 0.0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            on_exhausted=broken_observer,
        )

    assert raised.value is error


async def test_records_retry_limit_exhaustion() -> None:
    events: list[RetryExhaustedEvent] = []

    async def operation() -> None:
        raise ProviderError(503)

    with pytest.raises(ProviderError):
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(1, 0.25, 0.25),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            status_code=lambda error: error.status,
            sleep=lambda _: asyncio.sleep(0),
            jitter=lambda _low, high: high,
            on_exhausted=events.append,
        )

    assert len(events) == 1
    assert events[0].attempts == 2
    assert events[0].retries == 1
    assert events[0].cumulative_backoff_seconds == 0.25
    assert events[0].status_code == 503
    assert events[0].reason == "retry_limit"


async def test_records_non_retryable_exhaustion() -> None:
    events: list[RetryExhaustedEvent] = []

    async def operation() -> None:
        raise ProviderError(400)

    with pytest.raises(ProviderError):
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(4, 0.0, 0.0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: False,
            status_code=lambda error: error.status,
            on_exhausted=events.append,
        )

    assert events[0].attempts == 1
    assert events[0].retries == 0
    assert events[0].status_code == 400
    assert events[0].reason == "non_retryable"


async def test_end_to_end_deadline_cancels_in_flight_attempt() -> None:
    events: list[RetryExhaustedEvent] = []

    async def operation() -> None:
        await asyncio.Event().wait()

    with pytest.raises(RetryDeadlineExceeded) as raised:
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(4, 0.0, 0.0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            total_timeout_seconds=0.01,
            on_exhausted=events.append,
        )

    assert raised.value.event.reason == "deadline"
    assert raised.value.event.attempts == 1
    assert events == [raised.value.event]


async def test_deadline_spans_completed_backoff_and_multiple_attempts() -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError(503)
        await asyncio.Event().wait()

    with pytest.raises(RetryDeadlineExceeded) as raised:
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(3, 0.01, 0.01),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            total_timeout_seconds=0.04,
            jitter=lambda _low, high: high,
        )

    assert calls == 2
    assert raised.value.event.attempts == 2
    assert raised.value.event.cumulative_backoff_seconds == pytest.approx(0.01)


async def test_retry_budget_stops_retry_storm() -> None:
    calls = 0
    events: list[RetryExhaustedEvent] = []
    budget = RetryBudget(capacity=1, refill_rate_per_second=0)

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise ProviderError(503)

    with pytest.raises(ProviderError):
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(4, 0.0, 0.0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            retry_budget=budget,
            sleep=lambda _: asyncio.sleep(0),
            on_exhausted=events.append,
        )

    assert calls == 2
    assert events[0].reason == "retry_budget"
    assert events[0].attempts == 2
    assert budget.available_tokens == 0


async def test_retry_after_longer_than_deadline_fails_without_sleeping() -> None:
    sleeps: list[float] = []
    events: list[RetryExhaustedEvent] = []

    async def operation() -> None:
        raise ProviderError(429)

    with pytest.raises(RetryDeadlineExceeded):
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(4, 0.0, 0.0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            status_code=lambda error: error.status,
            retry_after=lambda _: 10.0,
            total_timeout_seconds=1.0,
            sleep=lambda delay: _record_delay(sleeps, delay),
            on_exhausted=events.append,
        )

    assert sleeps == []
    assert events[0].reason == "deadline"
    assert events[0].status_code == 429


async def test_retry_budget_refills_over_time() -> None:
    now = [10.0]
    budget = RetryBudget(2, 0.5, clock=lambda: now[0])

    assert budget.try_acquire()
    assert budget.try_acquire()
    assert not budget.try_acquire()
    now[0] += 2.0
    assert budget.try_acquire()


async def test_shared_retry_budget_caps_concurrent_retry_attempts() -> None:
    budget = RetryBudget(capacity=2, refill_rate_per_second=0)
    all_first_attempts_started = asyncio.Event()
    first_attempts = 0
    backoff_events: list[BackoffEvent] = []
    retry_events: list[RetryEvent] = []
    exhaustion_events: list[RetryExhaustedEvent] = []
    call_counts = [0] * 5

    async def run_request(index: int) -> str:
        async def operation() -> str:
            nonlocal first_attempts
            call_counts[index] += 1
            if call_counts[index] == 1:
                first_attempts += 1
                if first_attempts == len(call_counts):
                    all_first_attempts_started.set()
                await all_first_attempts_started.wait()
                raise ProviderError(503)
            return "ok"

        return await retry_with_backoff(
            operation,
            policy=RetryPolicy(1, 0.25, 0.25),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            retry_budget=budget,
            sleep=lambda _: asyncio.sleep(0),
            jitter=lambda _low, high: high,
            on_backoff=backoff_events.append,
            on_retry=retry_events.append,
            on_exhausted=exhaustion_events.append,
        )

    results = await asyncio.gather(
        *(run_request(index) for index in range(5)),
        return_exceptions=True,
    )

    assert results.count("ok") == 2
    assert sum(isinstance(result, ProviderError) for result in results) == 3
    assert sum(call_counts) == 7
    assert len(backoff_events) == 5
    assert sum(event.delay_seconds for event in backoff_events) == 1.25
    assert len(retry_events) == 2
    assert [event.reason for event in exhaustion_events] == [
        "retry_budget",
        "retry_budget",
        "retry_budget",
    ]


@pytest.mark.parametrize(
    ("capacity", "refill_rate", "message"),
    (
        (0, 1.0, "capacity"),
        (1, -1.0, "refill_rate_per_second"),
        (1, float("nan"), "refill_rate_per_second"),
    ),
)
async def test_retry_budget_rejects_invalid_configuration(
    capacity: int, refill_rate: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RetryBudget(capacity, refill_rate)


async def test_parses_provider_retry_after_headers() -> None:
    seconds_error = SimpleNamespace(
        response=SimpleNamespace(headers={"Retry-After": "2.5"})
    )
    milliseconds_error = SimpleNamespace(
        response=SimpleNamespace(headers={"retry-after-ms": "750"})
    )

    assert retry_after_seconds(seconds_error) == 2.5
    assert retry_after_seconds(milliseconds_error) == 0.75


async def test_retry_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        RetryPolicy(-1, 0.5, 2.0)
    with pytest.raises(ValueError, match="base_delay_seconds"):
        RetryPolicy(1, -0.5, 2.0)
    with pytest.raises(ValueError, match="max_delay_seconds"):
        RetryPolicy(1, 2.0, 0.5)
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="base_delay_seconds"):
            RetryPolicy(1, non_finite, 2.0)
        with pytest.raises(ValueError, match="max_delay_seconds"):
            RetryPolicy(1, 0.5, non_finite)


async def test_retry_policy_caps_extreme_retry_index_without_overflow() -> None:
    policy = RetryPolicy(10_000, 0.5, 2.0)

    assert policy.delay_for(10_000, jitter=lambda _low, high: high) == 2.0


async def _record_delay(delays: list[float], delay: float) -> None:
    delays.append(delay)
