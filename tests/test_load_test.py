from __future__ import annotations

import asyncio

import pytest

from llm import LLM
from load_test import LoadTestConfig, SyntheticProvider, Workload, run_load_test

pytestmark = pytest.mark.asyncio


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
            return LLM.SimpleResponse(
                answer=f"{system_prompt}: {question}",
                input_tokens=4,
                output_tokens=2,
            )
        finally:
            self.active -= 1


async def test_load_test_enforces_concurrency_and_reports_failures() -> None:
    provider = FakeProvider(fail_on_call=5)
    summary = await run_load_test(
        provider,
        Workload("system", ("one", "two")),
        LoadTestConfig(
            requests=12,
            concurrency=3,
            requests_per_second=0,
            temperature=0.2,
        ),
    )

    assert provider.max_active == 3
    assert summary["requests"] == 12
    assert summary["successes"] == 11
    assert summary["failures"] == 1
    assert summary["failure_modes"] == {"http_503": 1}
    assert summary["observed_input_tokens"] == 44
    assert summary["observed_output_tokens"] == 22
    assert summary["provider"] == {"provider": "FakeProvider"}
    assert summary["workload"]["question_count"] == 2
    assert len(summary["workload"]["sha256"]) == 64
    assert summary["runtime"]["dependencies"]["google-genai"] == "2.13.0"
    assert summary["retries"] is None
    assert summary["warmup"] == {
        "requests": 0,
        "successes": 0,
        "failures": 0,
        "failure_modes": {},
    }


async def test_cancellation_cleans_up_in_flight_workers() -> None:
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
            LoadTestConfig(
                requests=10,
                concurrency=3,
                requests_per_second=0,
                temperature=0.0,
            ),
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.active == 0


async def test_fatal_worker_failure_interrupts_request_scheduling() -> None:
    class FatalWorkerError(BaseException):
        pass

    class BrokenProvider(FakeProvider):
        async def ask_generic_question(
            self, system_prompt: str, question: str, temperature: float
        ) -> LLM.SimpleResponse:
            raise FatalWorkerError()

    with pytest.raises(FatalWorkerError):
        await asyncio.wait_for(
            run_load_test(
                BrokenProvider(),
                Workload("system", ("one",)),
                LoadTestConfig(
                    requests=10,
                    concurrency=2,
                    requests_per_second=0.1,
                    temperature=0.0,
                ),
            ),
            timeout=0.5,
        )


async def test_synthetic_provider_identifies_results_as_non_provider() -> None:
    provider = SyntheticProvider(parallelism=4, delay_seconds=0)
    summary = await run_load_test(
        provider,
        Workload("system", ("one",)),
        LoadTestConfig(
            requests=3,
            concurrency=2,
            requests_per_second=0,
            temperature=0.0,
        ),
    )

    assert summary["failures"] == 0
    assert summary["provider"]["provider"] == "SyntheticProvider"
    assert "does not measure an external LLM provider" in summary["provider"]["warning"]
    assert summary["service_latency_ms"]["p95"] > 0
    assert summary["end_to_end_latency_ms"]["p95"] >= summary[
        "service_latency_ms"
    ]["p95"]


async def test_warmup_requests_are_excluded_from_results() -> None:
    provider = FakeProvider()
    summary = await run_load_test(
        provider,
        Workload("system", ("one",)),
        LoadTestConfig(
            requests=5,
            concurrency=2,
            requests_per_second=0,
            temperature=0.0,
            warmup_requests=2,
        ),
    )

    assert provider.calls == 7
    assert summary["requests"] == 5
    assert summary["successes"] == 5
    assert summary["warmup"] == {
        "requests": 2,
        "successes": 2,
        "failures": 0,
        "failure_modes": {},
    }
