from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from llm import (
    BackoffEvent,
    LLM,
    LLMResponseError,
    ProviderOptions,
    RetryEvent,
    RetryExhaustedEvent,
    add_provider_arguments,
    create_provider,
    status_code_from_error,
)


@dataclass(frozen=True)
class Workload:
    system_prompt: str
    questions: tuple[str, ...]
    source: str | None = None

    @classmethod
    def load(cls, path: Path) -> Workload:
        document = json.loads(path.read_text(encoding="utf-8"))
        system_prompt = document.get("system_prompt")
        questions = document.get("questions")
        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValueError("workload system_prompt must be a non-empty string")
        if not isinstance(questions, list) or not questions:
            raise ValueError("workload questions must be a non-empty list")
        if any(not isinstance(question, str) or not question for question in questions):
            raise ValueError("every workload question must be a non-empty string")
        return cls(
            system_prompt=system_prompt,
            questions=tuple(questions),
            source=path.name,
        )

    def metadata(self) -> dict[str, object]:
        # Record a fingerprint instead of prompts so results are comparable without
        # duplicating potentially sensitive workload content into every artifact.
        serialized = json.dumps(
            {
                "system_prompt": self.system_prompt,
                "questions": self.questions,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return {
            "source": self.source,
            "sha256": hashlib.sha256(serialized).hexdigest(),
            "question_count": len(self.questions),
        }


@dataclass(frozen=True)
class LoadTestConfig:
    requests: int
    concurrency: int
    requests_per_second: float
    temperature: float
    warmup_requests: int = 0
    max_pending_requests: int | None = None

    def __post_init__(self) -> None:
        if self.requests < 1:
            raise ValueError("requests must be at least 1")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if (
            not math.isfinite(self.requests_per_second)
            or self.requests_per_second < 0
        ):
            raise ValueError("requests_per_second must be finite and non-negative")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must not be negative")
        if self.max_pending_requests is not None and self.max_pending_requests < 1:
            raise ValueError("max_pending_requests must be at least 1")

    @property
    def pending_request_limit(self) -> int:
        """Bound queued work while allowing a small open-loop arrival burst."""
        return self.max_pending_requests or self.concurrency * 4


@dataclass(frozen=True)
class RequestSample:
    success: bool
    service_latency_seconds: float
    end_to_end_latency_seconds: float
    queue_delay_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    error_type: str | None = None
    status_code: int | None = None


@dataclass(frozen=True)
class _PhaseResult:
    samples: list[RequestSample]
    queue_capacity: int
    max_queue_depth: int


class RetryCounter:
    """Aggregates retry telemetry; only counts are read, so no events are kept.

    A long throttled run can produce one event per retry; storing aggregates
    keeps memory constant regardless of run length.
    """

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._retry_count = 0
        self._retries_by_status: Counter[str] = Counter()
        self._cumulative_backoff_seconds = 0.0
        self._exhaustion_count = 0
        self._exhaustions_by_reason: Counter[str] = Counter()
        self._exhaustions_by_status: Counter[str] = Counter()

    def record_backoff(self, event: BackoffEvent) -> None:
        self._cumulative_backoff_seconds += event.delay_seconds

    def record(self, event: RetryEvent) -> None:
        self._retry_count += 1
        self._retries_by_status[_status_key(event)] += 1

    def record_exhaustion(self, event: RetryExhaustedEvent) -> None:
        self._exhaustion_count += 1
        self._exhaustions_by_reason[event.reason] += 1
        self._exhaustions_by_status[_status_key(event)] += 1

    def reset(self) -> None:
        self._reset_state()

    @property
    def count(self) -> int:
        return self._retry_count

    @property
    def cumulative_backoff_seconds(self) -> float:
        return self._cumulative_backoff_seconds

    @property
    def exhaustion_count(self) -> int:
        return self._exhaustion_count

    @property
    def exhaustions_by_reason(self) -> dict[str, int]:
        return dict(self._exhaustions_by_reason)

    @property
    def exhaustions_by_status(self) -> dict[str, int]:
        return dict(self._exhaustions_by_status)

    @property
    def by_status(self) -> dict[str, int]:
        return dict(self._retries_by_status)


def _status_key(event: RetryEvent | RetryExhaustedEvent) -> str:
    if event.status_code is not None:
        return str(event.status_code)
    return event.error_type


class SyntheticProvider(LLM):
    """Cheap local backend for validating the harness, never provider capacity."""

    def __init__(self, parallelism: int, delay_seconds: float = 0.001) -> None:
        self._parallelism = parallelism
        self._delay_seconds = delay_seconds

    def parallelism(self) -> int:
        return self._parallelism

    def metadata(self) -> dict[str, object]:
        return {
            "provider": "SyntheticProvider",
            "delay_seconds": self._delay_seconds,
            "parallelism": self._parallelism,
            "warning": "This run does not measure an external LLM provider.",
        }

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        await asyncio.sleep(self._delay_seconds)
        return LLM.SimpleResponse(answer="ok", input_tokens=8, output_tokens=2)


async def run_load_test(
    provider: LLM,
    workload: Workload,
    config: LoadTestConfig,
    retry_counter: RetryCounter | None = None,
) -> dict[str, Any]:
    warmup_phase = _PhaseResult(
        samples=[],
        queue_capacity=config.pending_request_limit,
        max_queue_depth=0,
    )
    if config.warmup_requests:
        warmup_phase = await _run_phase(
            provider,
            workload,
            requests=config.warmup_requests,
            concurrency=min(config.concurrency, config.warmup_requests),
            requests_per_second=0,
            temperature=config.temperature,
            max_pending_requests=config.pending_request_limit,
        )
        if retry_counter is not None:
            # Warmup establishes connections but must not contaminate measured retries.
            retry_counter.reset()

    started_at = datetime.now(timezone.utc)
    phase_started = time.perf_counter()
    measured_phase = await _run_phase(
        provider,
        workload,
        requests=config.requests,
        concurrency=config.concurrency,
        requests_per_second=config.requests_per_second,
        temperature=config.temperature,
        max_pending_requests=config.pending_request_limit,
    )
    elapsed_seconds = time.perf_counter() - phase_started

    summary = _summarize(measured_phase.samples, elapsed_seconds)
    summary.update(
        {
            "started_at": started_at.isoformat(),
            "config": asdict(config),
            "provider": provider.metadata(),
            "workload": workload.metadata(),
            "runtime": _runtime_metadata(),
            "queue_capacity": measured_phase.queue_capacity,
            "max_queue_depth": measured_phase.max_queue_depth,
            "warmup": _phase_summary(warmup_phase),
            **_retry_summary(retry_counter, len(measured_phase.samples)),
        }
    )
    return summary


async def _run_phase(
    provider: LLM,
    workload: Workload,
    *,
    requests: int,
    concurrency: int,
    requests_per_second: float,
    temperature: float,
    max_pending_requests: int,
) -> _PhaseResult:
    queue: asyncio.Queue[tuple[int, float] | None] = asyncio.Queue(
        maxsize=max_pending_requests
    )
    samples: list[RequestSample] = []
    max_queue_depth = 0

    async def worker() -> None:
        while True:
            work = await queue.get()
            if work is None:
                queue.task_done()
                return

            index, scheduled_at = work
            request_started = time.perf_counter()
            try:
                response = await provider.ask_generic_question(
                    workload.system_prompt,
                    workload.questions[index % len(workload.questions)],
                    temperature,
                )
                request_finished = time.perf_counter()
                samples.append(
                    RequestSample(
                        success=True,
                        service_latency_seconds=request_finished - request_started,
                        end_to_end_latency_seconds=request_finished - scheduled_at,
                        queue_delay_seconds=request_started - scheduled_at,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                    )
                )
            except Exception as error:
                request_finished = time.perf_counter()
                billed_input, billed_output = _billed_tokens(error)
                samples.append(
                    RequestSample(
                        success=False,
                        service_latency_seconds=request_finished - request_started,
                        end_to_end_latency_seconds=request_finished - scheduled_at,
                        queue_delay_seconds=request_started - scheduled_at,
                        input_tokens=billed_input,
                        output_tokens=billed_output,
                        error_type=type(error).__name__,
                        status_code=status_code_from_error(error),
                    )
                )
            finally:
                queue.task_done()

    async def produce() -> None:
        nonlocal max_queue_depth
        # Schedule against absolute deadlines. This preserves the offered arrival rate
        # even when service latency rises. A bounded queue applies backpressure while
        # scheduled timestamps preserve the resulting overload as queue delay.
        phase_started = time.perf_counter()
        for index in range(requests):
            if requests_per_second:
                scheduled_at = phase_started + (index / requests_per_second)
                await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
            else:
                scheduled_at = phase_started
            await queue.put((index, scheduled_at))
            max_queue_depth = max(max_queue_depth, queue.qsize())

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    producer = asyncio.create_task(produce())
    join_task: asyncio.Task[None] | None = None
    try:
        # A worker should not finish before the producer. Monitoring both prevents a
        # fatal worker error from leaving request scheduling alive with no consumers.
        done, _ = await asyncio.wait(
            [producer, *workers], return_when=asyncio.FIRST_COMPLETED
        )
        if producer not in done:
            next(iter(done)).result()
            raise RuntimeError("load-test worker stopped while scheduling requests")
        await producer

        join_task = asyncio.create_task(queue.join())
        # Workers normally remain alive until sentinels are added after all work drains.
        # If one exits first, surface its exception rather than hanging on queue.join().
        done, _ = await asyncio.wait(
            [join_task, *workers], return_when=asyncio.FIRST_COMPLETED
        )
        if join_task not in done:
            next(iter(done)).result()
            raise RuntimeError("load-test worker stopped before completing the queue")
        await join_task

        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
    finally:
        # Cancellation can occur during pacing, queue draining, or provider I/O. Always
        # reap every task so repeated load phases do not leak background workers.
        if not producer.done():
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        if join_task is not None and not join_task.done():
            join_task.cancel()
            await asyncio.gather(join_task, return_exceptions=True)
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    return _PhaseResult(
        samples=samples,
        queue_capacity=max_pending_requests,
        max_queue_depth=max_queue_depth,
    )


def _summarize(
    samples: Sequence[RequestSample], elapsed_seconds: float
) -> dict[str, Any]:
    successes = [sample for sample in samples if sample.success]
    failures = [sample for sample in samples if not sample.success]
    request_count = len(samples)
    success_count = len(successes)
    failure_modes = Counter(_failure_key(sample) for sample in failures)

    return {
        "elapsed_seconds": round(elapsed_seconds, 6),
        "requests": request_count,
        "successes": success_count,
        "failures": len(failures),
        "success_rate": success_count / request_count if request_count else 0.0,
        "attempted_requests_per_second": (
            request_count / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "successful_requests_per_second": (
            success_count / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "service_latency_ms": _latency_summary(
            [sample.service_latency_seconds for sample in samples]
        ),
        "end_to_end_latency_ms": _latency_summary(
            [sample.end_to_end_latency_seconds for sample in samples]
        ),
        "queue_delay_ms": _latency_summary(
            [sample.queue_delay_seconds for sample in samples]
        ),
        "observed_input_tokens": sum(sample.input_tokens for sample in samples),
        "observed_output_tokens": sum(sample.output_tokens for sample in samples),
        # A provider may bill a request that times out locally; only provider-side
        # billing telemetry can close that observability gap.
        "token_usage_scope": (
            "Responses with usage metadata only; this is a billing lower bound "
            "that can exclude timed-out or failed attempts."
        ),
        "failure_modes": dict(sorted(failure_modes.items())),
    }


def _phase_counts(samples: Sequence[RequestSample]) -> dict[str, Any]:
    failures = [sample for sample in samples if not sample.success]
    failure_modes = Counter(_failure_key(sample) for sample in failures)
    return {
        "requests": len(samples),
        "successes": len(samples) - len(failures),
        "failures": len(failures),
        "failure_modes": dict(sorted(failure_modes.items())),
    }


def _phase_summary(phase: _PhaseResult) -> dict[str, Any]:
    return {
        **_phase_counts(phase.samples),
        "queue_capacity": phase.queue_capacity,
        "max_queue_depth": phase.max_queue_depth,
    }


def _retry_summary(
    retry_counter: RetryCounter | None, request_count: int
) -> dict[str, Any]:
    if retry_counter is None:
        return {
            "retries": None,
            "provider_attempts": None,
            "retry_backoff_seconds": None,
            "retry_exhaustions": None,
            "retry_exhaustions_by_reason": None,
            "retry_exhaustions_by_status": None,
            "retries_by_status": None,
        }
    return {
        "retries": retry_counter.count,
        "provider_attempts": request_count + retry_counter.count,
        "retry_backoff_seconds": retry_counter.cumulative_backoff_seconds,
        "retry_exhaustions": retry_counter.exhaustion_count,
        "retry_exhaustions_by_reason": retry_counter.exhaustions_by_reason,
        "retry_exhaustions_by_status": retry_counter.exhaustions_by_status,
        "retries_by_status": retry_counter.by_status,
    }


def _latency_summary(values_seconds: Sequence[float]) -> dict[str, float]:
    values_ms = sorted(value * 1_000 for value in values_seconds)
    if not values_ms:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": values_ms[0],
        "p50": _percentile(values_ms, 0.50),
        "p95": _percentile(values_ms, 0.95),
        "p99": _percentile(values_ms, 0.99),
        "max": values_ms[-1],
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[index]


def _failure_key(sample: RequestSample) -> str:
    if sample.status_code is not None:
        return f"http_{sample.status_code}"
    return sample.error_type or "unknown"


def _billed_tokens(error: BaseException) -> tuple[int, int]:
    # Only the declared response-error contract carries billing telemetry.
    if isinstance(error, LLMResponseError):
        return error.input_tokens, error.output_tokens
    return 0, 0


def _runtime_metadata() -> dict[str, object]:
    # Dependency versions are provider-owned metadata (see LLM.metadata), so
    # the shared harness records only the interpreter and platform.
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an open-loop load test against an LLM provider."
    )
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument(
        "--rps",
        type=float,
        default=10.0,
        help="Offered requests/sec; use 0 to enqueue all requests immediately.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Maximum in-flight requests (defaults to the provider suggestion).",
    )
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument(
        "--max-pending-requests",
        type=int,
        help="Maximum queued arrivals (defaults to four times concurrency).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Exercise only the local harness; does not call an LLM provider.",
    )
    add_provider_arguments(parser)
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path(__file__).with_name("load_test_workload.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    workload = Workload.load(args.workload)
    retry_counter = RetryCounter()

    # `is None` checks keep an explicit 0 flowing into config validation instead
    # of silently substituting the provider default.
    if args.synthetic:
        provider: LLM = SyntheticProvider(
            args.concurrency if args.concurrency is not None else 128
        )
    else:
        try:
            provider = create_provider(
                args.provider,
                ProviderOptions(
                    model=args.model,
                    max_retries=args.max_retries,
                    max_output_tokens=args.max_output_tokens,
                    thinking_budget=args.thinking_budget,
                    on_backoff=retry_counter.record_backoff,
                    on_retry=retry_counter.record,
                    on_exhausted=retry_counter.record_exhaustion,
                ),
            )
        except ValueError as error:
            # Configuration mistakes should fail as usage errors before any
            # billable request, not as tracebacks.
            print(f"error: {error}", file=sys.stderr)
            return 2

    try:
        config = LoadTestConfig(
            requests=args.requests,
            concurrency=(
                args.concurrency
                if args.concurrency is not None
                else provider.parallelism()
            ),
            requests_per_second=args.rps,
            temperature=args.temperature,
            warmup_requests=args.warmup_requests,
            max_pending_requests=args.max_pending_requests,
        )
        summary = await run_load_test(provider, workload, config, retry_counter)
    finally:
        await provider.close()

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return _exit_code(summary)


def _exit_code(summary: dict[str, Any]) -> int:
    # Warmup failures stay visible in the report but must not abort a campaign
    # whose measured phase is healthy.
    return 1 if summary["failures"] else 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
