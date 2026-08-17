from __future__ import annotations

import asyncio

import pytest

from llm import LLM, RetryEvent
from load_test import (
    LoadTestConfig,
    RetryCounter,
    Workload,
    _exit_code,
    threshold_breaches,
    run_load_test,
)


class ServiceUnavailable(RuntimeError):
    code = 503


class FakeProvider(LLM):
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.fail_on_call = fail_on_call

    def parallelism(self) -> int:
        return 3

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        self.calls += 1
        call = self.calls
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.001)
            if call == self.fail_on_call:
                raise ServiceUnavailable("unavailable")
            return LLM.SimpleResponse(f"{system_prompt}: {question}", 4, 2)
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_reports_load_failures_tokens_warmup_and_concurrency() -> None:
    provider = FakeProvider(fail_on_call=5)
    report = await run_load_test(
        provider,
        Workload("system", ("one", "two")),
        LoadTestConfig(12, 3, 0, 0.2, warmup_requests=2),
    )

    assert provider.calls == 14
    assert provider.max_active == 3
    assert report["requests"] == 12
    assert report["successes"] == 11
    assert report["failure_modes"] == {"http_503": 1}
    failure = report["failure_samples"][0]
    assert failure["success"] is False
    assert failure["service_seconds"] == pytest.approx(0.001, rel=0.5)
    assert failure["end_to_end_seconds"] == pytest.approx(0.001, rel=0.5)
    assert failure["scheduler_seconds"] == pytest.approx(0, abs=0.001)
    assert failure["backpressure_seconds"] == pytest.approx(0, abs=0.001)
    assert failure["queue_seconds"] == pytest.approx(0, abs=0.001)
    assert failure["error_type"] == "ServiceUnavailable"
    assert failure["status_code"] == 503
    assert report["failure_samples_truncated"] == 0
    assert report["realized_arrival_rps"] > 0
    assert report["max_active_requests"] == 3
    assert report["observed_input_tokens"] == 44
    assert report["observed_output_tokens"] == 22
    assert report["warmup"]["requests"] == 2
    assert report["warmup"]["failures"] == 0


@pytest.mark.asyncio
async def test_warmup_retries_do_not_leak_into_measured_telemetry() -> None:
    counter = RetryCounter()

    class RetryingProvider(FakeProvider):
        async def ask_generic_question(
            self, system_prompt: str, question: str, temperature: float
        ) -> LLM.SimpleResponse:
            counter.record(RetryEvent(1, 0, 503, "ServiceUnavailable"))
            return LLM.SimpleResponse("ok", 1, 1)

    report = await run_load_test(
        RetryingProvider(),
        Workload("system", ("one",)),
        LoadTestConfig(2, 1, 0, 0, warmup_requests=1),
        counter,
    )
    assert report["retries"] == 2
    assert report["provider_attempts"] == 4


@pytest.mark.asyncio
async def test_pending_arrivals_are_bounded() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class GatedProvider(FakeProvider):
        async def ask_generic_question(
            self, system_prompt: str, question: str, temperature: float
        ) -> LLM.SimpleResponse:
            self.calls += 1
            started.set()
            await release.wait()
            return LLM.SimpleResponse("ok", 1, 1)

    provider = GatedProvider()
    task = asyncio.create_task(
        run_load_test(
            provider,
            Workload("system", ("one",)),
            LoadTestConfig(20, 1, 0, 0, max_pending_requests=2),
        )
    )
    await started.wait()
    await asyncio.sleep(0)
    assert provider.calls == 1
    release.set()

    report = await task
    assert report["queue_capacity"] == 2
    assert report["max_queue_depth"] == 2


@pytest.mark.asyncio
async def test_cancellation_cleans_up_workers() -> None:
    started = asyncio.Event()

    class BlockingProvider(FakeProvider):
        async def ask_generic_question(
            self, system_prompt: str, question: str, temperature: float
        ) -> LLM.SimpleResponse:
            self.active += 1
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1

    provider = BlockingProvider()
    task = asyncio.create_task(
        run_load_test(
            provider,
            Workload("system", ("one",)),
            LoadTestConfig(10, 3, 0, 0),
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.active == 0


@pytest.mark.asyncio
async def test_emit_samples_exposes_raw_samples_and_wall_clock_arrivals() -> None:
    report = await run_load_test(
        FakeProvider(),
        Workload("system", ("one",)),
        LoadTestConfig(5, 2, 0, 0, emit_samples=True),
    )
    assert len(report["samples"]) == 5
    assert all(sample["service_seconds"] > 0 for sample in report["samples"])
    assert report["arrival_first_unix"] <= report["arrival_last_unix"]

    baseline = await run_load_test(
        FakeProvider(),
        Workload("system", ("one",)),
        LoadTestConfig(2, 1, 0, 0),
    )
    assert "samples" not in baseline


def test_exit_code_applies_failure_and_latency_thresholds() -> None:
    report = {
        "requests": 300,
        "failures": 1,
        "service_latency_ms": {"p95": 1200.0},
        "end_to_end_latency_ms": {"p95": 1500.0},
    }
    assert _exit_code(report) == 1
    assert _exit_code(report, max_failure_rate=0.01) == 0
    assert _exit_code(report, max_failure_rate=0.01, max_service_p95_ms=1000) == 1
    assert threshold_breaches(report, 0.01, 2000, 1000) == ["end_to_end_p95"]
