"""Bounded open-loop load harness for the Gemini production-readiness experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from llm import (
    BackoffEvent,
    Gemini,
    GeminiConfig,
    LLM,
    LLMResponseError,
    RetryEvent,
    RetryExhaustedEvent,
    status_code_from_error,
)

FAILURE_SAMPLE_LIMIT = 20


@dataclass(frozen=True)
class Workload:
    """Fixed prompt set cycled round-robin across a run's requests."""

    system_prompt: str
    questions: tuple[str, ...]
    source: str | None = None

    @classmethod
    def load(cls, path: Path) -> Workload:
        """Parse a workload JSON file, rejecting empty or non-string fields."""
        document = json.loads(path.read_text(encoding="utf-8"))
        prompt = document.get("system_prompt")
        questions = document.get("questions")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("workload system_prompt must be a non-empty string")
        if not isinstance(questions, list) or not questions or not all(
            isinstance(question, str) and question for question in questions
        ):
            raise ValueError("workload questions must be a non-empty string list")
        return cls(prompt, tuple(questions), path.name)

    def metadata(self) -> dict[str, object]:
        """Identify the workload by content hash so reports from different
        runs are comparable only when they exercised identical prompts."""
        content = json.dumps(
            {"system_prompt": self.system_prompt, "questions": self.questions},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return {
            "source": self.source,
            "sha256": hashlib.sha256(content).hexdigest(),
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
    # Include raw per-request samples in the report so a multi-process
    # orchestrator can compute true pooled percentiles; percentile summaries
    # from separate shards cannot be merged after the fact.
    emit_samples: bool = False

    def __post_init__(self) -> None:
        if self.requests < 1 or self.concurrency < 1:
            raise ValueError("requests and concurrency must be positive")
        if not math.isfinite(self.requests_per_second) or self.requests_per_second < 0:
            raise ValueError("requests_per_second must be finite and non-negative")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative")
        if self.max_pending_requests is not None and self.max_pending_requests < 1:
            raise ValueError("max_pending_requests must be positive")

    @property
    def pending_limit(self) -> int:
        # Bounds accepted-but-unstarted arrivals; the default gives workers
        # a small buffer without letting an open-loop backlog grow unbounded.
        return self.max_pending_requests or self.concurrency * 4


@dataclass(frozen=True)
class RequestSample:
    """One request's timing, split along the open-loop pipeline.

    Each request passes through: ideal scheduled instant -> producer wakes
    (scheduler_seconds late) -> bounded-slot acquire (backpressure_seconds)
    -> worker pickup (queue_seconds) -> provider call (service_seconds).
    end_to_end_seconds spans ideal instant to completion, so it equals
    service latency only when the client pipeline added no delay — that
    separation is what distinguishes provider saturation from generator
    saturation in the reports.
    """

    success: bool
    service_seconds: float
    end_to_end_seconds: float
    scheduler_seconds: float
    backpressure_seconds: float
    queue_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    error_type: str | None = None
    status_code: int | None = None


@dataclass(frozen=True)
class _Phase:
    samples: list[RequestSample]
    elapsed_seconds: float
    producer_elapsed_seconds: float
    arrival_span_seconds: float
    queue_capacity: int
    max_queue_depth: int
    max_active_requests: int
    # Wall-clock arrival bounds let an orchestrator compute the overlap
    # window across shard processes, whose monotonic clocks are not comparable.
    first_enqueued_unix: float | None = None
    last_enqueued_unix: float | None = None


class RetryCounter:
    """Aggregates provider retry callbacks into report-ready totals."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.retries = 0
        self.backoff_seconds = 0.0
        self.by_status = Counter()
        self.exhaustions = 0
        self.exhaustions_by_reason = Counter()
        self.exhaustions_by_status = Counter()

    def record(self, event: RetryEvent) -> None:
        self.retries += 1
        self.by_status[_status_key(event.status_code)] += 1

    def record_backoff(self, event: BackoffEvent) -> None:
        self.backoff_seconds += event.delay_seconds

    def record_exhaustion(self, event: RetryExhaustedEvent) -> None:
        self.exhaustions += 1
        self.exhaustions_by_reason[event.reason] += 1
        self.exhaustions_by_status[_status_key(event.status_code)] += 1

    def summary(self, requests: int) -> dict[str, Any]:
        return {
            "retries": self.retries,
            "provider_attempts": requests + self.retries,
            "retry_backoff_seconds": self.backoff_seconds,
            "retries_by_status": dict(sorted(self.by_status.items())),
            "retry_exhaustions": self.exhaustions,
            "retry_exhaustions_by_reason": dict(
                sorted(self.exhaustions_by_reason.items())
            ),
            "retry_exhaustions_by_status": dict(
                sorted(self.exhaustions_by_status.items())
            ),
        }


class SyntheticProvider(LLM):
    """In-process ~1 ms provider for exercising the harness without spend."""

    def parallelism(self) -> int:
        return 128

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        await asyncio.sleep(0.001)
        return LLM.SimpleResponse("synthetic", 4, 2)


async def run_load_test(
    provider: LLM,
    workload: Workload,
    config: LoadTestConfig,
    retry_counter: RetryCounter | None = None,
) -> dict[str, Any]:
    """Run an unpaced warmup then the paced measured phase; return the report.

    Only the measured phase contributes samples and retry telemetry; warmup
    exists to absorb cold authentication and connection setup.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    warmup = await _run_phase(
        provider, workload, config, config.warmup_requests, paced=False
    )
    # Retries triggered during warmup must not leak into measured telemetry.
    if retry_counter is not None:
        retry_counter.reset()
    measured = await _run_phase(provider, workload, config, config.requests, paced=True)
    summary = _summarize(measured, config, retry_counter)
    summary.update(
        {
            "started_at": started_at,
            "provider": provider.metadata(),
            "workload": workload.metadata(),
            "config": asdict(config),
            "warmup": _counts(warmup),
        }
    )
    return summary


async def _run_phase(
    provider: LLM,
    workload: Workload,
    config: LoadTestConfig,
    requests: int,
    *,
    paced: bool,
) -> _Phase:
    """Drive one open-loop phase: a producer emits arrivals on the ideal
    schedule (or as fast as slots allow when unpaced), a bounded semaphore
    plus queue applies backpressure, and worker tasks call the provider.

    Arrival pacing never waits for completions — that is what keeps offered
    rate independent of provider latency — so the only intentional producer
    stalls are the pending-slot bound.
    """
    if requests == 0:
        return _Phase([], 0.0, 0.0, 0.0, config.pending_limit, 0, 0)

    loop = asyncio.get_running_loop()
    phase_started = loop.time()
    wall_clock_offset = time.time() - phase_started
    queue: asyncio.Queue[tuple[int, float, float, float]] = asyncio.Queue(
        config.pending_limit
    )
    pending_slots = asyncio.Semaphore(config.pending_limit)
    samples: list[RequestSample] = []
    max_queue_depth = 0
    active_requests = 0
    max_active_requests = 0
    first_enqueued_at: float | None = None
    last_enqueued_at: float | None = None

    async def worker() -> None:
        nonlocal active_requests, max_active_requests
        while True:
            index, scheduled_at, ready_at, enqueued_at = await queue.get()
            pending_slots.release()
            request_started = loop.time()
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            try:
                response = await provider.ask_generic_question(
                    workload.system_prompt,
                    workload.questions[index % len(workload.questions)],
                    config.temperature,
                )
                ended = loop.time()
                samples.append(
                    RequestSample(
                        True,
                        ended - request_started,
                        ended - scheduled_at,
                        ready_at - scheduled_at,
                        enqueued_at - ready_at,
                        request_started - enqueued_at,
                        response.input_tokens,
                        response.output_tokens,
                        response.thought_tokens,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                ended = loop.time()
                input_tokens = (
                    error.input_tokens if isinstance(error, LLMResponseError) else 0
                )
                output_tokens = (
                    error.output_tokens if isinstance(error, LLMResponseError) else 0
                )
                samples.append(
                    RequestSample(
                        False,
                        ended - request_started,
                        ended - scheduled_at,
                        ready_at - scheduled_at,
                        enqueued_at - ready_at,
                        request_started - enqueued_at,
                        input_tokens,
                        output_tokens,
                        error_type=type(error).__name__,
                        status_code=status_code_from_error(error),
                    )
                )
            finally:
                active_requests -= 1
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(config.concurrency)]
    try:
        for index in range(requests):
            scheduled_at = phase_started
            if paced and config.requests_per_second:
                scheduled_at += index / config.requests_per_second
                await asyncio.sleep(max(0.0, scheduled_at - loop.time()))
            ready_at = loop.time()
            await pending_slots.acquire()
            enqueued_at = loop.time()
            if first_enqueued_at is None:
                first_enqueued_at = enqueued_at
            last_enqueued_at = enqueued_at
            queue.put_nowait((index, scheduled_at, ready_at, enqueued_at))
            max_queue_depth = max(max_queue_depth, queue.qsize())
        producer_elapsed_seconds = loop.time() - phase_started
        await queue.join()
    finally:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    arrival_span_seconds = (
        last_enqueued_at - first_enqueued_at
        if first_enqueued_at is not None and last_enqueued_at is not None
        else 0.0
    )
    return _Phase(
        samples,
        loop.time() - phase_started,
        producer_elapsed_seconds,
        arrival_span_seconds,
        config.pending_limit,
        max_queue_depth,
        max_active_requests,
        first_enqueued_unix=(
            wall_clock_offset + first_enqueued_at
            if first_enqueued_at is not None
            else None
        ),
        last_enqueued_unix=(
            wall_clock_offset + last_enqueued_at
            if last_enqueued_at is not None
            else None
        ),
    )


def _summarize(
    phase: _Phase,
    config: LoadTestConfig,
    retries: RetryCounter | None,
) -> dict[str, Any]:
    """Reduce a measured phase to the JSON report body: counts, latency
    percentiles per pipeline stage, token totals, and bounded failure
    samples (structured fields only — never provider message text)."""
    samples = phase.samples
    successes = sum(sample.success for sample in samples)
    failed_samples = [sample for sample in samples if not sample.success]
    failure_modes = Counter(_failure_key(sample) for sample in failed_samples)
    summary: dict[str, Any] = {
        "requests": len(samples),
        "successes": successes,
        "failures": len(samples) - successes,
        "failure_modes": dict(sorted(failure_modes.items())),
        "elapsed_seconds": phase.elapsed_seconds,
        "offered_rps": config.requests_per_second,
        "successful_rps": successes / phase.elapsed_seconds,
        # n arrivals define n-1 intervals across the first-to-last span.
        "realized_arrival_rps": (
            (len(samples) - 1) / phase.arrival_span_seconds
            if len(samples) > 1 and phase.arrival_span_seconds > 0
            else 0.0
        ),
        "producer_elapsed_seconds": phase.producer_elapsed_seconds,
        "arrival_span_seconds": phase.arrival_span_seconds,
        "arrival_first_unix": phase.first_enqueued_unix,
        "arrival_last_unix": phase.last_enqueued_unix,
        "queue_capacity": phase.queue_capacity,
        "max_queue_depth": phase.max_queue_depth,
        "max_active_requests": phase.max_active_requests,
        "service_latency_ms": latency_summary_ms(
            sample.service_seconds for sample in samples
        ),
        "end_to_end_latency_ms": latency_summary_ms(
            sample.end_to_end_seconds for sample in samples
        ),
        "scheduler_lag_ms": latency_summary_ms(
            sample.scheduler_seconds for sample in samples
        ),
        "backpressure_delay_ms": latency_summary_ms(
            sample.backpressure_seconds for sample in samples
        ),
        "queue_wait_ms": latency_summary_ms(
            sample.queue_seconds for sample in samples
        ),
        "observed_input_tokens": sum(sample.input_tokens for sample in samples),
        "observed_output_tokens": sum(sample.output_tokens for sample in samples),
        "observed_thought_tokens": sum(sample.thought_tokens for sample in samples),
        "failure_samples": [
            asdict(sample) for sample in failed_samples[:FAILURE_SAMPLE_LIMIT]
        ],
        "failure_samples_truncated": max(
            0,
            len(failed_samples) - FAILURE_SAMPLE_LIMIT,
        ),
    }
    if config.emit_samples:
        summary["samples"] = [asdict(sample) for sample in samples]
    summary.update((retries or RetryCounter()).summary(len(samples)))
    return summary


def _counts(phase: _Phase) -> dict[str, Any]:
    """Warmup gets outcome counts only; its latencies are cold-start noise."""
    failures = Counter(
        _failure_key(sample) for sample in phase.samples if not sample.success
    )
    return {
        "requests": len(phase.samples),
        "successes": sum(sample.success for sample in phase.samples),
        "failures": sum(not sample.success for sample in phase.samples),
        "failure_modes": dict(sorted(failures.items())),
        "queue_capacity": phase.queue_capacity,
        "max_queue_depth": phase.max_queue_depth,
        "max_active_requests": phase.max_active_requests,
    }


def _failure_key(sample: RequestSample) -> str:
    return f"http_{sample.status_code}" if sample.status_code else str(sample.error_type)


def latency_summary_ms(values: Iterable[float]) -> dict[str, float]:
    """Summarize second-valued latencies as millisecond percentiles.

    Public because the multi-process orchestrator recomputes pooled
    percentiles from merged shard samples with the same definition.
    """
    ordered = sorted(value * 1_000 for value in values)
    if not ordered:
        return {key: 0.0 for key in ("p50", "p95", "p99", "max")}
    return {
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Linear interpolation between the two nearest ranks of a sorted list."""
    position = (len(values) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _status_key(status: int | None) -> str:
    return str(status) if status is not None else "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=Path("load_test_workload.json"))
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--rps", type=float, required=True)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-pending-requests", type=int)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--model")
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--emit-samples", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-failure-rate", type=float, default=0)
    parser.add_argument("--max-service-p95-ms", type=float)
    parser.add_argument("--max-end-to-end-p95-ms", type=float)
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    retry_counter = RetryCounter()
    if args.synthetic:
        provider: LLM = SyntheticProvider()
    else:
        config = GeminiConfig.from_env()
        overrides = {}
        for name in ("model", "max_retries", "max_output_tokens", "thinking_budget"):
            if getattr(args, name) is not None:
                overrides[name] = getattr(args, name)
        provider = Gemini(
            replace(config, **overrides),
            on_backoff=retry_counter.record_backoff,
            on_retry=retry_counter.record,
            on_exhausted=retry_counter.record_exhaustion,
        )

    config = LoadTestConfig(
        requests=args.requests,
        concurrency=args.concurrency or provider.parallelism(),
        requests_per_second=args.rps,
        temperature=args.temperature,
        warmup_requests=args.warmup_requests,
        max_pending_requests=args.max_pending_requests,
        emit_samples=args.emit_samples,
    )
    try:
        report = await run_load_test(
            provider, Workload.load(args.workload), config, retry_counter
        )
    finally:
        await provider.close()

    report["runtime"] = {"python": platform.python_version()}
    report["acceptance"] = {
        "max_failure_rate": args.max_failure_rate,
        "max_service_p95_ms": args.max_service_p95_ms,
        "max_end_to_end_p95_ms": args.max_end_to_end_p95_ms,
        "breaches": threshold_breaches(
            report,
            args.max_failure_rate,
            args.max_service_p95_ms,
            args.max_end_to_end_p95_ms,
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return _exit_code(
        report,
        args.max_failure_rate,
        args.max_service_p95_ms,
        args.max_end_to_end_p95_ms,
    )


def _exit_code(
    report: dict[str, Any],
    max_failure_rate: float = 0,
    max_service_p95_ms: float | None = None,
    max_end_to_end_p95_ms: float | None = None,
) -> int:
    return int(
        bool(
            threshold_breaches(
                report,
                max_failure_rate,
                max_service_p95_ms,
                max_end_to_end_p95_ms,
            )
        )
    )


def threshold_breaches(
    report: dict[str, Any],
    max_failure_rate: float = 0,
    max_service_p95_ms: float | None = None,
    max_end_to_end_p95_ms: float | None = None,
) -> list[str]:
    """Name each acceptance rail the report breaches; None disables a rail.

    Public because the multi-process orchestrator applies the same rails to
    its pooled report.
    """
    failure_rate = report["failures"] / report["requests"]
    breaches = []
    if failure_rate > max_failure_rate:
        breaches.append("failure_rate")
    if (
        max_service_p95_ms is not None
        and report["service_latency_ms"]["p95"] > max_service_p95_ms
    ):
        breaches.append("service_p95")
    if (
        max_end_to_end_p95_ms is not None
        and report["end_to_end_latency_ms"]["p95"] > max_end_to_end_p95_ms
    ):
        breaches.append("end_to_end_p95")
    return breaches


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
