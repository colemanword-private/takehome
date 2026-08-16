from __future__ import annotations

import asyncio

import pytest

from llm import RetryEvent, RetryPolicy, retry_with_backoff

pytestmark = pytest.mark.asyncio


class ProviderError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"provider error {status}")
        self.status = status


async def test_retries_with_full_jitter_and_reports_event() -> None:
    outcomes: list[str | BaseException] = [ProviderError(429), "ok"]
    delays: list[float] = []
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
        on_retry=events.append,
    )

    assert result == "ok"
    assert delays == [0.25]
    assert events == [
        RetryEvent(
            attempt=1,
            delay_seconds=0.25,
            status_code=429,
            error_type="ProviderError",
        )
    ]


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

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise ProviderError(503)

    async def cancel_sleep(_: float) -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(2, 0.5, 2.0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            sleep=cancel_sleep,
        )

    assert calls == 1


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
