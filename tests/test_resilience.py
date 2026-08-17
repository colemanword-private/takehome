from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors

from conftest import FakeClient, config, response
from llm import (
    Gemini,
    RetryBudget,
    RetryDeadlineExceeded,
    RetryPolicy,
    retry_after_seconds,
    retry_with_backoff,
)
from load_test import LoadTestConfig, RetryCounter, Workload, run_load_test


class ProviderError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"provider error {status}")
        self.status = status


@pytest.mark.asyncio
async def test_retry_after_floors_jitter_and_reports_telemetry() -> None:
    outcomes: list[str | BaseException] = [ProviderError(429), "ok"]
    delays = []
    backoffs = []
    retries = []

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
        retry_after=lambda _: 1.5,
        sleep=sleep,
        jitter=lambda low, high: (low + high) / 2,
        on_backoff=backoffs.append,
        on_retry=retries.append,
    )

    assert result == "ok"
    assert delays == [1.5]
    assert backoffs[0].cumulative_backoff_seconds == 1.5
    assert retries[0].status_code == 429
    assert retries[0].retry_after_seconds == 1.5
    assert RetryPolicy(2, 0.5, 2).delay_for(0, lambda low, high: high / 2) == 0.25


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries", [0, 2])
async def test_stops_after_configured_retries(max_retries: int) -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise ProviderError(503)

    with pytest.raises(ProviderError):
        await retry_with_backoff(
            operation,
            policy=RetryPolicy(max_retries, 0, 0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
        )
    assert calls == max_retries + 1


@pytest.mark.asyncio
async def test_concurrent_requests_share_retry_budget() -> None:
    budget = RetryBudget(1, 0)
    attempts = 0

    async def request() -> str:
        local_attempts = 0

        async def operation() -> str:
            nonlocal attempts, local_attempts
            attempts += 1
            local_attempts += 1
            if local_attempts == 1:
                raise ProviderError(503)
            return "ok"

        return await retry_with_backoff(
            operation,
            policy=RetryPolicy(1, 0, 0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: True,
            retry_budget=budget,
        )

    results = await asyncio.gather(request(), request(), return_exceptions=True)
    assert attempts == 3
    assert sum(result == "ok" for result in results) == 1
    assert sum(isinstance(result, ProviderError) for result in results) == 1


@pytest.mark.asyncio
async def test_non_retryable_errors_and_cancellation_are_not_retried() -> None:
    calls = 0

    async def rejected() -> None:
        nonlocal calls
        calls += 1
        raise ProviderError(400)

    with pytest.raises(ProviderError):
        await retry_with_backoff(
            rejected,
            policy=RetryPolicy(2, 0, 0),
            handled_errors=(ProviderError,),
            is_retryable=lambda _: False,
        )
    assert calls == 1

    async def cancelled() -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await retry_with_backoff(
            cancelled,
            policy=RetryPolicy(2, 0, 0),
            handled_errors=(BaseException,),
            is_retryable=lambda _: True,
        )


@pytest.mark.asyncio
async def test_deadline_spans_backoff_and_in_flight_attempt() -> None:
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


def test_parses_retry_after_headers() -> None:
    seconds = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "2.5"}))
    milliseconds = SimpleNamespace(
        response=SimpleNamespace(headers={"retry-after-ms": "750"})
    )
    assert retry_after_seconds(seconds) == 2.5
    assert retry_after_seconds(milliseconds) == 0.75


def _rate_limit_error() -> errors.ClientError:
    error = errors.ClientError(429, {"error": {"message": "quota exceeded"}})
    error.response = SimpleNamespace(headers={"retry-after": "3"})
    return error


def _load_config() -> LoadTestConfig:
    return LoadTestConfig(1, 1, 0, 0)


@pytest.mark.asyncio
async def test_stalled_gemini_request_is_bounded() -> None:
    async def stall() -> SimpleNamespace:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    provider = Gemini(
        config(max_retries=0, timeout_seconds=0.05), client=FakeClient([stall])
    )
    report = await run_load_test(provider, Workload("system", ("one",)), _load_config())

    assert report["failure_modes"] == {"TimeoutError": 1}
    assert report["service_latency_ms"]["max"] < 1_000
    await provider.close()


@pytest.mark.asyncio
async def test_mixed_faults_flow_through_provider_and_load_report() -> None:
    retry_counter = RetryCounter()
    unavailable = errors.ServerError(503, {"error": {"message": "overloaded"}})
    provider = Gemini(
        config(retry_max_delay_seconds=8),
        client=FakeClient([unavailable, _rate_limit_error(), response()]),
        sleep=lambda _: asyncio.sleep(0),
        jitter=lambda low, high: low,
        on_backoff=retry_counter.record_backoff,
        on_retry=retry_counter.record,
        on_exhausted=retry_counter.record_exhaustion,
    )
    report = await run_load_test(
        provider, Workload("system", ("one",)), _load_config(), retry_counter
    )

    assert report["successes"] == 1
    assert report["provider_attempts"] == 3
    assert report["retries_by_status"] == {"429": 1, "503": 1}
    assert report["retry_backoff_seconds"] == 3
    await provider.close()
