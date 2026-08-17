"""Offline fault injection through the real provider, retry engine, and harness.

The live campaign observed zero provider failures, which left the failure
paths unexercised end-to-end. These tests script a stalled request and mixed
429/503 faults through the real Gemini provider so deadline bounding and retry
telemetry are demonstrated without live cost.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors

from llm import Gemini, GeminiConfig
from load_test import LoadTestConfig, RetryCounter, Workload, run_load_test

pytestmark = pytest.mark.asyncio


class FakeModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes

    async def generate_content(self, **_: Any) -> Any:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome


class FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.aio = SimpleNamespace(
            models=FakeModels(outcomes), aclose=self._aclose
        )

    async def _aclose(self) -> None:
        return None

    def close(self) -> None:
        return None


def config(**overrides: Any) -> GeminiConfig:
    values: dict[str, Any] = {
        "project": "project-id",
        "max_retries": 2,
        "retry_base_delay_seconds": 0.5,
        "retry_max_delay_seconds": 8.0,
    }
    values.update(overrides)
    return GeminiConfig(**values)


def response(
    text: str | None = "answer",
    *,
    prompt_tokens: int = 5,
    candidate_tokens: int = 6,
    thought_tokens: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidate_tokens,
            thoughts_token_count=thought_tokens,
            total_token_count=prompt_tokens + candidate_tokens + thought_tokens,
        ),
        candidates=[],
        prompt_feedback=None,
    )


def rate_limit_error(retry_after: str | None = None) -> errors.ClientError:
    error = errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "quota exceeded",
            }
        },
    )
    if retry_after is not None:
        error.response = SimpleNamespace(headers={"retry-after": retry_after})
    return error


def server_error() -> errors.ServerError:
    return errors.ServerError(
        503,
        {"error": {"code": 503, "status": "UNAVAILABLE", "message": "overloaded"}},
    )


WORKLOAD = Workload("system", ("one",))


def load_config(requests: int) -> LoadTestConfig:
    return LoadTestConfig(
        requests=requests,
        concurrency=1,
        requests_per_second=0,
        temperature=0.0,
    )


async def test_stalled_request_is_bounded_by_the_per_attempt_timeout() -> None:
    async def stall() -> SimpleNamespace:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    provider = Gemini(
        config(max_retries=0, timeout_seconds=0.05),
        client=FakeClient([stall]),
    )

    summary = await run_load_test(provider, WORKLOAD, load_config(1))

    assert summary["failures"] == 1
    assert summary["failure_modes"] == {"TimeoutError": 1}
    assert summary["service_latency_ms"]["max"] < 1_000
    await provider.close()


async def test_retry_telemetry_is_accurate_under_mixed_faults() -> None:
    retry_counter = RetryCounter()
    provider = Gemini(
        config(),
        client=FakeClient(
            [server_error(), rate_limit_error(retry_after="3"), response()]
        ),
        sleep=lambda _: asyncio.sleep(0),
        jitter=lambda low, high: low,
        on_backoff=retry_counter.record_backoff,
        on_retry=retry_counter.record,
        on_exhausted=retry_counter.record_exhaustion,
    )

    summary = await run_load_test(provider, WORKLOAD, load_config(1), retry_counter)

    assert summary["successes"] == 1
    assert summary["retries"] == 2
    assert summary["provider_attempts"] == 3
    assert summary["retries_by_status"] == {"429": 1, "503": 1}
    # First backoff draws zero jitter; the second is the server's 3s directive.
    assert summary["retry_backoff_seconds"] == 3.0
    assert summary["retry_exhaustions"] == 0
    await provider.close()
