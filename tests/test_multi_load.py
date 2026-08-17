from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import multi_load
from multi_load import merge_shard_reports, shard_breaches, shard_requests

PROJECT_DIR = Path(__file__).parent.parent


def sample(service: float) -> dict[str, float]:
    return {
        "success": True,
        "service_seconds": service,
        "end_to_end_seconds": service + 0.01,
        "scheduler_seconds": 0.001,
        "backpressure_seconds": 0.0,
        "queue_seconds": 0.0005,
    }


def shard_report(
    services: list[float],
    *,
    first_unix: float,
    last_unix: float,
    failures: int = 0,
    workload_sha: str = "workload-sha",
    model: str = "gemini-2.5-flash",
    scheduler_p95_ms: float = 2.0,
) -> dict[str, Any]:
    requests = len(services)
    return {
        "started_at": "2026-08-17T00:00:00+00:00",
        "requests": requests,
        "successes": requests - failures,
        "failures": failures,
        "failure_modes": {"http_503": failures} if failures else {},
        "offered_rps": 50.0,
        "realized_arrival_rps": 50.0,
        "arrival_span_seconds": last_unix - first_unix,
        "arrival_first_unix": first_unix,
        "arrival_last_unix": last_unix,
        "elapsed_seconds": last_unix - first_unix + 1.0,
        "max_active_requests": 40,
        "observed_input_tokens": 100,
        "observed_output_tokens": 50,
        "observed_thought_tokens": 0,
        "retries": 1,
        "provider_attempts": requests + 1,
        "retry_backoff_seconds": 0.25,
        "retries_by_status": {"503": 1},
        "retry_exhaustions": 0,
        "retry_exhaustions_by_reason": {},
        "retry_exhaustions_by_status": {},
        "scheduler_lag_ms": {"p50": 1.0, "p95": scheduler_p95_ms, "p99": 3.0, "max": 4.0},
        "service_latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0},
        "provider": {"model": model},
        "workload": {"sha256": workload_sha},
        "samples": [sample(service) for service in services],
    }


def test_shard_requests_splits_with_remainder() -> None:
    assert shard_requests(10, 4) == [3, 3, 2, 2]
    assert shard_requests(8, 4) == [2, 2, 2, 2]
    with pytest.raises(ValueError):
        shard_requests(3, 4)


def test_merges_pooled_percentiles_arrivals_and_counters() -> None:
    merged = merge_shard_reports(
        [
            shard_report([0.1, 0.2, 0.3], first_unix=1000.0, last_unix=1030.0),
            shard_report(
                [0.4, 0.5, 0.6], first_unix=1000.5, last_unix=1030.5, failures=1
            ),
        ]
    )

    assert merged["requests"] == 6
    assert merged["failures"] == 1
    assert merged["failure_modes"] == {"http_503": 1}
    assert merged["retries"] == 2
    assert merged["retries_by_status"] == {"503": 2}
    assert merged["offered_rps"] == 100.0
    # Pooled median must come from the union of raw samples, not from
    # averaging shard percentiles: sorted ms are 100..600, so p50 is 350.
    assert merged["service_latency_ms"]["p50"] == pytest.approx(350.0)
    assert merged["arrival_span_seconds"] == pytest.approx(30.5)
    assert merged["realized_arrival_rps"] == pytest.approx(5 / 30.5)
    assert merged["arrival_overlap_seconds"] == pytest.approx(29.5)
    assert merged["arrival_overlap_fraction"] == pytest.approx(29.5 / 30.5)
    assert len(merged["shards"]) == 2
    assert merged["shards"][1]["failures"] == 1


def test_rejects_mismatched_or_incomplete_shards() -> None:
    baseline = shard_report([0.1], first_unix=1.0, last_unix=2.0)
    with pytest.raises(ValueError, match="different workloads"):
        merge_shard_reports(
            [baseline, shard_report([0.1], first_unix=1.0, last_unix=2.0, workload_sha="other")]
        )
    with pytest.raises(ValueError, match="different models"):
        merge_shard_reports(
            [baseline, shard_report([0.1], first_unix=1.0, last_unix=2.0, model="other")]
        )
    missing_samples = shard_report([0.1], first_unix=1.0, last_unix=2.0)
    del missing_samples["samples"]
    with pytest.raises(ValueError, match="missing raw samples"):
        merge_shard_reports([baseline, missing_samples])
    missing_arrival = shard_report([0.1], first_unix=1.0, last_unix=2.0)
    missing_arrival["arrival_first_unix"] = None
    with pytest.raises(ValueError, match="missing arrival timestamps"):
        merge_shard_reports([baseline, missing_arrival])


def test_shard_scheduler_lag_marks_stage_client_dirty() -> None:
    clean = merge_shard_reports(
        [shard_report([0.1], first_unix=1.0, last_unix=2.0)] * 2
    )
    assert shard_breaches(clean, 10.0) == []
    dirty = merge_shard_reports(
        [
            shard_report([0.1], first_unix=1.0, last_unix=2.0),
            shard_report([0.1], first_unix=1.0, last_unix=2.0, scheduler_p95_ms=80.0),
        ]
    )
    assert shard_breaches(dirty, 10.0) == ["shard_scheduler_lag"]


@pytest.mark.asyncio
async def test_refuses_to_overwrite_existing_stage_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "merged.json"
    args = SimpleNamespace(output=output, requests=8, shards=2)

    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="never overwritten"):
        await multi_load.run_shards(args)

    output.unlink()
    (tmp_path / "merged-shards").mkdir()
    with pytest.raises(FileExistsError, match="never overwritten"):
        await multi_load.run_shards(args)


@pytest.mark.asyncio
async def test_failed_shard_terminates_billable_siblings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        async def wait(self) -> int | None:
            return self.returncode

    siblings: list[FakeProcess] = []

    async def fake_run_shard(
        args: SimpleNamespace,
        index: int,
        requests: int,
        output: Path,
        processes: list,
    ) -> Path:
        if index == 0:
            # Fail only after every sibling is running, so the test proves
            # live processes get reaped rather than never started.
            while len(siblings) < 3:
                await asyncio.sleep(0)
            raise RuntimeError("shard 0 failed with exit code 2")
        process = FakeProcess()
        processes.append(process)
        siblings.append(process)
        await asyncio.Event().wait()
        return output

    monkeypatch.setattr(multi_load, "_run_shard", fake_run_shard)
    args = SimpleNamespace(output=tmp_path / "merged.json", requests=40, shards=4)

    with pytest.raises(RuntimeError, match="shard 0"):
        await multi_load.run_shards(args)

    assert len(siblings) == 3
    assert all(process.terminated for process in siblings)


def test_synthetic_two_shard_run_produces_merged_report(tmp_path: Path) -> None:
    output = tmp_path / "merged.json"
    command = [
        sys.executable,
        "multi_load.py",
        "--synthetic",
        "--requests", "200",
        "--total-rps", "2000",
        "--shards", "2",
        "--shard-concurrency", "16",
        "--warmup-requests", "0",
        "--output", str(output),
        # This test verifies orchestration, not host timing.
        "--max-shard-scheduler-p95-ms", "10000",
    ]
    result = subprocess.run(
        command, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["requests"] == 200
    assert merged["failures"] == 0
    assert len(merged["shards"]) == 2
    assert merged["acceptance"]["breaches"] == []
    assert 0.0 < merged["arrival_overlap_fraction"] <= 1.0
    shard_files = sorted((tmp_path / "merged-shards").glob("shard-*.json"))
    assert len(shard_files) == 2
    for path in shard_files:
        shard = json.loads(path.read_text(encoding="utf-8"))
        assert len(shard["samples"]) == shard["requests"] == 100

    # A rerun against the same paths must refuse before spending anything.
    rerun = subprocess.run(
        command, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=120
    )
    assert rerun.returncode == 2
    assert "never overwritten" in rerun.stderr
