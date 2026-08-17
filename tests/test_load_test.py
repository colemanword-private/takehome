from __future__ import annotations

import asyncio

import pytest

from llm import (
    LLM,
    RetryBudget,
    RetryPolicy,
    retry_with_backoff,
)
from load_test import (
    LoadTestConfig,
    RetryCounter,
    SyntheticProvider,
    Workload,
    run_load_test,
)

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
    # Dependency versions are provider-owned metadata; the shared runtime block
    # records only the interpreter and platform.
    assert set(summary["runtime"]) == {"python", "platform"}
    assert summary["retries"] is None
    assert summary["queue_capacity"] == 12
    assert summary["max_queue_depth"] <= summary["queue_capacity"]
    assert summary["warmup"] == {
        "requests": 0,
        "successes": 0,
        "failures": 0,
        "failure_modes": {},
        "queue_capacity": 12,
        "max_queue_depth": 0,
    }


async def test_failed_requests_count_tokens_only_from_declared_response_errors() -> None:
    from llm import LLMResponseError

    class BilledFailureProvider(FakeProvider):
        async def ask_generic_question(
            self, system_prompt: str, question: str, temperature: float
        ) -> LLM.SimpleResponse:
            raise LLMResponseError("no usable text", input_tokens=7, output_tokens=3)

    class UndeclaredBilledError(RuntimeError):
        input_tokens = 7
        output_tokens = 3

    class UndeclaredFailureProvider(FakeProvider):
        async def ask_generic_question(
            self, system_prompt: str, question: str, temperature: float
        ) -> LLM.SimpleResponse:
            raise UndeclaredBilledError("boom")

    config = LoadTestConfig(
        requests=1, concurrency=1, requests_per_second=0, temperature=0.0
    )
    workload = Workload("system", ("one",))

    declared = await run_load_test(BilledFailureProvider(), workload, config)
    undeclared = await run_load_test(UndeclaredFailureProvider(), workload, config)

    assert declared["observed_input_tokens"] == 7
    assert declared["observed_output_tokens"] == 3
    # Attributes on an undeclared exception type are not billing telemetry.
    assert undeclared["observed_input_tokens"] == 0
    assert undeclared["observed_output_tokens"] == 0


async def test_failure_modes_classify_real_genai_rate_limit() -> None:
    from google.genai import errors

    class RateLimitedProvider(FakeProvider):
        async def ask_generic_question(
            self, system_prompt: str, question: str, temperature: float
        ) -> LLM.SimpleResponse:
            raise errors.ClientError(
                429,
                {
                    "error": {
                        "code": 429,
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "quota exceeded",
                    }
                },
            )

    summary = await run_load_test(
        RateLimitedProvider(),
        Workload("system", ("one",)),
        LoadTestConfig(
            requests=1,
            concurrency=1,
            requests_per_second=0,
            temperature=0.0,
        ),
    )

    assert summary["failure_modes"] == {"http_429": 1}


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
        "queue_capacity": 8,
        "max_queue_depth": 2,
    }


async def test_pending_request_queue_applies_bounded_backpressure() -> None:
    first_request_started = asyncio.Event()
    release_requests = asyncio.Event()

    class GatedProvider(FakeProvider):
        async def ask_generic_question(
            self, system_prompt: str, question: str, temperature: float
        ) -> LLM.SimpleResponse:
            self.calls += 1
            first_request_started.set()
            await release_requests.wait()
            return LLM.SimpleResponse("ok", input_tokens=1, output_tokens=1)

    provider = GatedProvider()
    task = asyncio.create_task(
        run_load_test(
            provider,
            Workload("system", ("one",)),
            LoadTestConfig(
                requests=20,
                concurrency=1,
                requests_per_second=0,
                temperature=0.0,
                max_pending_requests=2,
            ),
        )
    )

    await first_request_started.wait()
    await asyncio.sleep(0)
    assert provider.calls == 1
    release_requests.set()
    summary = await task

    assert summary["requests"] == 20
    assert summary["queue_capacity"] == 2
    assert summary["max_queue_depth"] == 2


async def test_queue_limit_may_be_smaller_than_worker_concurrency() -> None:
    summary = await run_load_test(
        FakeProvider(),
        Workload("system", ("one",)),
        LoadTestConfig(
            requests=10,
            concurrency=4,
            requests_per_second=0,
            temperature=0.0,
            max_pending_requests=1,
        ),
    )

    assert summary["successes"] == 10
    assert summary["queue_capacity"] == 1
    assert summary["max_queue_depth"] == 1


async def test_load_report_includes_retry_attempt_and_exhaustion_telemetry() -> None:
    retry_counter = RetryCounter()

    class FaultingRetryProvider(LLM):
        def __init__(self) -> None:
            self.requests = 0
            self.provider_calls = 0
            self.retry_budget = RetryBudget(1, 0)

        def parallelism(self) -> int:
            return 1

        async def ask_generic_question(
            self, system_prompt: str, question: str, temperature: float
        ) -> LLM.SimpleResponse:
            self.requests += 1
            request_number = self.requests
            attempts = 0

            async def operation() -> LLM.SimpleResponse:
                nonlocal attempts
                attempts += 1
                self.provider_calls += 1
                if request_number == 2 or attempts == 1:
                    raise ServiceUnavailable("unavailable")
                return LLM.SimpleResponse("ok", input_tokens=1, output_tokens=1)

            return await retry_with_backoff(
                operation,
                policy=RetryPolicy(2, 0.25, 0.25),
                handled_errors=(ServiceUnavailable,),
                is_retryable=lambda _: True,
                status_code=lambda error: error.code,
                retry_budget=self.retry_budget,
                sleep=lambda _: asyncio.sleep(0),
                jitter=lambda _low, high: high,
                on_backoff=retry_counter.record_backoff,
                on_retry=retry_counter.record,
                on_exhausted=retry_counter.record_exhaustion,
            )

    provider = FaultingRetryProvider()

    summary = await run_load_test(
        provider,
        Workload("system", ("one",)),
        LoadTestConfig(
            requests=2,
            concurrency=1,
            requests_per_second=0,
            temperature=0.0,
        ),
        retry_counter,
    )

    assert provider.provider_calls == 3
    assert summary["successes"] == 1
    assert summary["failures"] == 1
    assert summary["retries"] == 1
    assert summary["provider_attempts"] == 3
    assert summary["retry_backoff_seconds"] == 0.5
    assert summary["retry_exhaustions"] == 1
    assert summary["retry_exhaustions_by_reason"] == {"retry_budget": 1}
    assert summary["retry_exhaustions_by_status"] == {"503": 1}


async def test_cli_reports_unsupported_control_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An unsupported control must exit with a clean usage error before any
    # billable work, not an unhandled traceback.
    import load_test as load_test_module

    monkeypatch.setattr(
        "sys.argv",
        [
            "load_test.py",
            "--provider",
            "together",
            "--model",
            "organization/model",
            "--thinking-budget",
            "0",
        ],
    )

    assert await load_test_module._main() == 2
    assert "thinking-budget" in capsys.readouterr().err


async def test_exit_code_ignores_warmup_failures_when_measured_phase_is_healthy() -> None:
    # A transient warmup blip must not abort a multi-stage campaign whose
    # measured phase succeeded; the warmup counts stay visible in the report.
    import load_test as load_test_module

    healthy_measured = {"failures": 0, "warmup": {"failures": 1}}
    failed_measured = {"failures": 1, "warmup": {"failures": 0}}

    assert load_test_module._exit_code(healthy_measured) == 0
    assert load_test_module._exit_code(failed_measured) == 1


async def test_exit_code_applies_configurable_abort_thresholds() -> None:
    import load_test as load_test_module

    tolerable = {
        "requests": 300,
        "failures": 1,
        "warmup": {"failures": 0},
        "service_latency_ms": {"p95": 1200.0},
    }
    breaching_rate = {**tolerable, "failures": 4}
    breaching_p95 = {**tolerable, "failures": 0}

    # Default: any measured failure aborts.
    assert load_test_module._exit_code(tolerable) == 1
    # A 1% threshold tolerates 1/300 but aborts at 4/300 (>1.3%).
    assert load_test_module._exit_code(tolerable, max_failure_rate=0.01) == 0
    assert load_test_module._exit_code(breaching_rate, max_failure_rate=0.01) == 1
    # The p95 threshold aborts independently of the failure rate.
    assert (
        load_test_module._exit_code(
            breaching_p95, max_failure_rate=0.01, max_service_p95_ms=5000.0
        )
        == 0
    )
    assert (
        load_test_module._exit_code(breaching_p95, max_service_p95_ms=1000.0) == 1
    )


async def test_cli_aborts_when_service_p95_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import load_test as load_test_module

    monkeypatch.setattr(
        "sys.argv",
        [
            "load_test.py",
            "--synthetic",
            "--requests",
            "5",
            "--rps",
            "0",
            "--concurrency",
            "2",
            "--warmup-requests",
            "0",
            "--max-service-p95-ms",
            "0.0001",
        ],
    )

    assert await load_test_module._main() == 1
    assert "abort:" in capsys.readouterr().err


async def test_cli_rejects_out_of_range_failure_rate_threshold(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import load_test as load_test_module

    monkeypatch.setattr(
        "sys.argv",
        ["load_test.py", "--synthetic", "--max-failure-rate", "1.5"],
    )

    with pytest.raises(SystemExit):
        await load_test_module._main()
    assert "between 0 and 1" in capsys.readouterr().err


async def test_cli_rejects_explicit_zero_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An explicit --concurrency 0 must be rejected, not silently replaced with
    # the provider default and run at full concurrency.
    import load_test as load_test_module

    monkeypatch.setattr(
        "sys.argv",
        [
            "load_test.py",
            "--synthetic",
            "--concurrency",
            "0",
            "--requests",
            "1",
            "--warmup-requests",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="concurrency"):
        await load_test_module._main()


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"requests_per_second": float("nan")}, "requests_per_second"),
        ({"requests_per_second": float("inf")}, "requests_per_second"),
        ({"max_pending_requests": 0}, "max_pending_requests"),
    ),
)
async def test_load_config_rejects_invalid_admission_values(
    overrides: dict[str, float | int], message: str
) -> None:
    values: dict[str, float | int | None] = {
        "requests": 1,
        "concurrency": 1,
        "requests_per_second": 1.0,
        "temperature": 0.0,
        "max_pending_requests": None,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        LoadTestConfig(**values)  # type: ignore[arg-type]
