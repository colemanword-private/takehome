"""Multi-process load orchestrator: N load_test.py shards, one pooled report.

A single Python event loop saturates between the 150 and 200 RPS targets, so
rates above that need several generator processes. Each shard stays the
tested single-process harness; this orchestrator only plans shard rates,
spawns the processes, and merges their raw samples. Percentiles are always
recomputed from pooled samples because per-shard percentile summaries cannot
be merged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from load_test import latency_summary_ms, threshold_breaches

LOAD_TEST = Path(__file__).with_name("load_test.py")

# Above this per-shard scheduler-lag p95, arrivals no longer tracked the ideal
# schedule and the stage measures the generator, not the provider.
DEFAULT_SHARD_SCHEDULER_P95_MS = 10.0

_MERGED_LATENCY_KEYS = {
    "service_latency_ms": "service_seconds",
    "end_to_end_latency_ms": "end_to_end_seconds",
    "scheduler_lag_ms": "scheduler_seconds",
    "backpressure_delay_ms": "backpressure_seconds",
    "queue_wait_ms": "queue_seconds",
}
_SUMMED_KEYS = (
    "requests",
    "successes",
    "failures",
    "observed_input_tokens",
    "observed_output_tokens",
    "observed_thought_tokens",
    "retries",
    "provider_attempts",
    "retry_backoff_seconds",
    "retry_exhaustions",
)
_MERGED_COUNTER_KEYS = (
    "failure_modes",
    "retries_by_status",
    "retry_exhaustions_by_reason",
    "retry_exhaustions_by_status",
)


def shard_requests(total_requests: int, shards: int) -> list[int]:
    """Split a request total across shards, front-loading the remainder."""
    if total_requests < shards:
        raise ValueError("need at least one request per shard")
    base, remainder = divmod(total_requests, shards)
    return [base + (1 if index < remainder else 0) for index in range(shards)]


def merge_shard_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool shard samples into one report shaped like a single-shard report."""
    if not reports:
        raise ValueError("no shard reports to merge")
    for key in ("sha256",):
        values = {report["workload"][key] for report in reports}
        if len(values) != 1:
            raise ValueError(f"shards ran different workloads: {values}")
    models = {report["provider"].get("model") for report in reports}
    if len(models) != 1:
        raise ValueError(f"shards ran different models: {models}")

    samples: list[dict[str, Any]] = []
    for report in reports:
        shard_samples = report.get("samples")
        if not isinstance(shard_samples, list):
            raise ValueError("shard report is missing raw samples")
        samples.extend(shard_samples)

    merged: dict[str, Any] = {
        key: sum(report[key] for report in reports) for key in _SUMMED_KEYS
    }
    for key in _MERGED_COUNTER_KEYS:
        counts: Counter[str] = Counter()
        for report in reports:
            counts.update(report[key])
        merged[key] = dict(sorted(counts.items()))
    for merged_key, sample_key in _MERGED_LATENCY_KEYS.items():
        merged[merged_key] = latency_summary_ms(
            sample[sample_key] for sample in samples
        )

    firsts = [report["arrival_first_unix"] for report in reports]
    lasts = [report["arrival_last_unix"] for report in reports]
    if any(value is None for value in firsts + lasts):
        raise ValueError("shard report is missing arrival timestamps")
    union_span = max(lasts) - min(firsts)
    overlap_span = max(0.0, min(lasts) - max(firsts))
    merged.update(
        {
            "started_at": min(report["started_at"] for report in reports),
            "offered_rps": sum(report["offered_rps"] for report in reports),
            "arrival_span_seconds": union_span,
            "realized_arrival_rps": (
                (merged["requests"] - 1) / union_span if union_span > 0 else 0.0
            ),
            # The pooled rate holds for the whole union window only if every
            # shard was emitting throughout; report how true that was.
            "arrival_overlap_seconds": overlap_span,
            "arrival_overlap_fraction": (
                overlap_span / union_span if union_span > 0 else 0.0
            ),
            "elapsed_seconds": max(
                report["elapsed_seconds"] for report in reports
            ),
            "provider": reports[0]["provider"],
            "workload": reports[0]["workload"],
            "shards": [_shard_summary(report) for report in reports],
        }
    )
    return merged


def shard_breaches(
    merged: dict[str, Any], max_shard_scheduler_p95_ms: float
) -> list[str]:
    """Flag the stage as client-dirty if any shard's scheduler lag shows the
    generator, not the provider, was the bottleneck."""
    dirty = any(
        shard["scheduler_lag_p95_ms"] > max_shard_scheduler_p95_ms
        for shard in merged["shards"]
    )
    return ["shard_scheduler_lag"] if dirty else []


def _shard_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Keep the per-shard fields the merged report and evidence rely on."""
    return {
        "offered_rps": report["offered_rps"],
        "requests": report["requests"],
        "successes": report["successes"],
        "failures": report["failures"],
        "realized_arrival_rps": report["realized_arrival_rps"],
        "arrival_span_seconds": report["arrival_span_seconds"],
        "scheduler_lag_p95_ms": report["scheduler_lag_ms"]["p95"],
        "service_p95_ms": report["service_latency_ms"]["p95"],
        "max_active_requests": report["max_active_requests"],
        "retries": report["retries"],
    }


async def _reap(processes: list[asyncio.subprocess.Process]) -> None:
    """Terminate and wait on any shard process that is still running.

    A failed or interrupted run must never leave sibling shards issuing
    billable requests with nobody supervising them.
    """
    survivors = [
        process for process in processes if process.returncode is None
    ]
    for process in survivors:
        process.terminate()
    for process in survivors:
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.kill()
            await process.wait()


async def run_shards(args: argparse.Namespace) -> list[Path]:
    """Run every shard, failing fast and reaping survivors on any failure."""
    # Raw shard reports (with samples) live in a sibling directory so the
    # evidence generator's flat report glob only sees the merged stage report.
    shard_dir = args.output.parent / f"{args.output.stem}-shards"
    if args.output.exists() or shard_dir.exists():
        raise FileExistsError(
            f"{args.output} or {shard_dir} already exists; stage artifacts "
            "are never overwritten so reruns cannot mix campaigns — use a "
            "fresh RUN_ID or remove the stale files first"
        )
    shard_dir.mkdir(parents=True)
    processes: list[asyncio.subprocess.Process] = []
    tasks = [
        asyncio.create_task(
            _run_shard(
                args, index, requests, shard_dir / f"shard-{index}.json", processes
            )
        )
        for index, requests in enumerate(
            shard_requests(args.requests, args.shards)
        )
    ]
    try:
        done, _pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            if task.exception() is not None:
                raise task.exception()
        return [task.result() for task in tasks]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _reap(processes)


async def _run_shard(
    args: argparse.Namespace,
    index: int,
    requests: int,
    output: Path,
    processes: list[asyncio.subprocess.Process],
) -> Path:
    """Spawn one load_test.py subprocess and wait for its report.

    The process is registered in `processes` before awaiting so a failure
    elsewhere can reap it; raises if the shard exits nonzero or writes no
    report.
    """
    command = [
        sys.executable,
        str(LOAD_TEST),
        "--workload", str(args.workload),
        "--requests", str(requests),
        "--rps", str(args.total_rps / args.shards),
        "--concurrency", str(args.shard_concurrency),
        "--warmup-requests", str(args.warmup_requests),
        "--temperature", str(args.temperature),
        "--emit-samples",
        # Shards never fail on thresholds; acceptance is judged on the pool.
        "--max-failure-rate", "1",
        "--output", str(output),
    ]
    if args.synthetic:
        command.append("--synthetic")
    for name in ("model", "max_retries", "max_output_tokens", "thinking_budget"):
        value = getattr(args, name)
        if value is not None:
            command.extend((f"--{name.replace('_', '-')}", str(value)))

    env = dict(os.environ)
    if not args.synthetic:
        # Size each shard's provider connection pool to its own worker count.
        env["GEMINI_PARALLELISM"] = str(args.shard_concurrency)

    process = await asyncio.create_subprocess_exec(
        *command, env=env, stdout=asyncio.subprocess.DEVNULL
    )
    processes.append(process)
    code = await process.wait()
    if code != 0 or not output.exists():
        raise RuntimeError(f"shard {index} failed with exit code {code}")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=Path("load_test_workload.json"))
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--total-rps", type=float, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--shard-concurrency", type=int, default=128)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--model")
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-failure-rate", type=float, default=0)
    parser.add_argument("--max-service-p95-ms", type=float)
    parser.add_argument("--max-end-to-end-p95-ms", type=float)
    parser.add_argument(
        "--max-shard-scheduler-p95-ms",
        type=float,
        default=DEFAULT_SHARD_SCHEDULER_P95_MS,
    )
    args = parser.parse_args()
    if args.shards < 1 or args.total_rps <= 0:
        parser.error("shards must be positive and total-rps must be > 0")
    return args


async def _main() -> int:
    """Exit 0 on a clean stage, 1 on an acceptance breach (report still
    written), 2 on operational failure — the Makefile stage loops stop on
    a breach but abort the whole campaign on a 2."""
    args = _parse_args()
    try:
        shard_paths = await run_shards(args)
        merged = merge_shard_reports(
            [
                json.loads(path.read_text(encoding="utf-8"))
                for path in shard_paths
            ]
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        # Operational failures exit 2 so the Makefile stage loops abort
        # instead of reading them as acceptance breaches (exit 1).
        print(f"multi_load: {error}", file=sys.stderr)
        return 2
    merged["config"] = {
        "requests": args.requests,
        "total_rps": args.total_rps,
        "shards": args.shards,
        "shard_concurrency": args.shard_concurrency,
        "warmup_requests": args.warmup_requests,
        "temperature": args.temperature,
    }
    breaches = threshold_breaches(
        merged,
        args.max_failure_rate,
        args.max_service_p95_ms,
        args.max_end_to_end_p95_ms,
    ) + shard_breaches(merged, args.max_shard_scheduler_p95_ms)
    merged["acceptance"] = {
        "max_failure_rate": args.max_failure_rate,
        "max_service_p95_ms": args.max_service_p95_ms,
        "max_end_to_end_p95_ms": args.max_end_to_end_p95_ms,
        "max_shard_scheduler_p95_ms": args.max_shard_scheduler_p95_ms,
        "breaches": breaches,
    }

    rendered = json.dumps(merged, indent=2, sort_keys=True)
    print(rendered)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return int(bool(breaches))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
