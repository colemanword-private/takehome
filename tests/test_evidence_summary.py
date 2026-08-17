from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from evidence_summary import build_summary, render_markdown

MODEL = "gemini-2.5-flash"
RUN_ID = "20260817T000000Z"


def load_report(
    *,
    offered_rps: float,
    requests: int,
    input_tokens: int,
    output_tokens: int,
    breaches: list[str] | None = None,
    shards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "offered_rps": offered_rps,
        "realized_arrival_rps": offered_rps,
        "requests": requests,
        "successes": requests,
        "failures": 0,
        "failure_modes": {},
        "retries": 3,
        "retries_by_status": {"504": 3},
        "retry_exhaustions": 0,
        "max_active_requests": 21,
        "config": {"requests": requests, "concurrency": 96},
        "observed_input_tokens": input_tokens,
        "observed_output_tokens": output_tokens,
        "observed_thought_tokens": 0,
        "service_latency_ms": {"p50": 590.0, "p95": 836.0, "p99": 1266.0, "max": 2000.0},
        "end_to_end_latency_ms": {"p50": 592.0, "p95": 838.0, "p99": 1270.0, "max": 2001.0},
        "scheduler_lag_ms": {"p50": 1.0, "p95": 2.82, "p99": 4.0, "max": 9.0},
        "provider": {
            "project": "secret-project",
            "model": MODEL,
            "max_output_tokens": 256,
            "thinking_budget": 0,
            "max_retries": 4,
            "timeout_seconds": 20.0,
            "request_deadline_seconds": 60.0,
        },
        "acceptance": {"breaches": breaches or []},
    }
    if shards is not None:
        report["shards"] = shards
    return report


def quality_report(
    *, thinking_budget: int | None, input_tokens: int, output_tokens: int
) -> dict[str, Any]:
    return {
        "cases": 10,
        "passed": 10,
        "failed": 0,
        "elapsed_seconds": 2.5,
        "observed_input_tokens": input_tokens,
        "observed_output_tokens": output_tokens,
        "observed_thought_tokens": output_tokens // 2,
        "provider": {
            "project": "secret-project",
            "thinking_budget": thinking_budget,
            "max_output_tokens": 256,
        },
    }


@pytest.fixture
def campaign_dir(tmp_path: Path) -> Path:
    load_dir = tmp_path / "load-results" / "gemini" / MODEL / RUN_ID
    eval_dir = tmp_path / "eval-results" / "gemini" / MODEL / RUN_ID
    load_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    reports = {
        load_dir / "operating-point-24rps.json": load_report(
            offered_rps=24, requests=1000, input_tokens=1_000_000, output_tokens=1_000_000
        ),
        load_dir / "bracket-200rps.json": load_report(
            offered_rps=200,
            requests=6000,
            input_tokens=300_000,
            output_tokens=600_000,
            breaches=["service_p95"],
            shards=[
                {"scheduler_lag_p95_ms": 2.0, "max_active_requests": 100},
                {"scheduler_lag_p95_ms": 5.5, "max_active_requests": 128},
            ],
        ),
        eval_dir / "quality-think-default-cap-256-r1.json": quality_report(
            thinking_budget=None, input_tokens=500_000, output_tokens=1_000_000
        ),
        eval_dir / "quality-think-0-cap-256-r1.json": quality_report(
            thinking_budget=0, input_tokens=500_000, output_tokens=100_000
        ),
    }
    for path, report in reports.items():
        path.write_text(json.dumps(report), encoding="utf-8")
    shards_dir = load_dir / "bracket-200rps-shards"
    shards_dir.mkdir()
    (shards_dir / "shard-0.json").write_text('{"samples": []}', encoding="utf-8")
    (shards_dir / "shard-1.json").write_text('{"samples": [1]}', encoding="utf-8")
    return tmp_path


def test_builds_compact_hashed_summary_with_cost(campaign_dir: Path) -> None:
    summary = build_summary(campaign_dir, RUN_ID, MODEL)

    bracket = next(
        row for row in summary["load_reports"] if row["file"] == "bracket-200rps.json"
    )
    source = (
        campaign_dir / "load-results" / "gemini" / MODEL / RUN_ID / "bracket-200rps.json"
    )
    assert bracket["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert bracket["breaches"] == ["service_p95"]
    assert bracket["shards"] == 2
    assert bracket["shard_scheduler_lag_p95_ms"] == 5.5
    assert bracket["shard_max_active_requests"] == 128
    assert bracket["max_output_tokens"] == 256
    assert bracket["timeout_seconds"] == 20.0
    assert bracket["retries_by_status"] == {"504": 3}
    assert bracket["config"] == {"requests": 6000, "concurrency": 96}
    shards_dir = (
        campaign_dir / "load-results" / "gemini" / MODEL / RUN_ID
        / "bracket-200rps-shards"
    )
    assert bracket["shard_report_sha256"] == {
        name: hashlib.sha256((shards_dir / name).read_bytes()).hexdigest()
        for name in ("shard-0.json", "shard-1.json")
    }

    cost = summary["cost"]
    # 1M input + 1M output at $0.30/$2.50 per million is $2.80 over 1k requests.
    assert cost["operating_point"]["usd_per_1k_requests"] == pytest.approx(2.8)
    assert cost["operating_point"]["usd_per_day_at_offered_rps"] == pytest.approx(
        2.8 / 1000 * 24 * 86_400
    )
    compare = cost["quality_thinking_comparison"]
    assert compare["default_thinking"]["usd_per_run"] == pytest.approx(2.65)
    assert compare["thinking_disabled"]["usd_per_run"] == pytest.approx(0.40)
    assert compare["cost_multiplier"] == pytest.approx(6.6)
    assert cost["campaign_total"]["requests"] == 7020
    # 2.3M input and 2.7M output tokens: 2.3 * $0.30 + 2.7 * $2.50 = $7.44.
    assert cost["campaign_total"]["usd"] == pytest.approx(7.44)

    # Committed evidence must never carry the project identifier.
    assert "secret-project" not in json.dumps(summary)


def test_renders_markdown_tables(campaign_dir: Path) -> None:
    summary = build_summary(campaign_dir, RUN_ID, MODEL)
    markdown = render_markdown(summary)
    assert "bracket-200rps.json" in markdown
    assert "service_p95" in markdown
    assert "$2.8/1k requests" in markdown
    assert "6.6x" in markdown
    assert "secret-project" not in markdown


def test_rejects_missing_campaign(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no reports found"):
        build_summary(tmp_path, RUN_ID, MODEL)


def test_rejects_shard_files_from_another_run(campaign_dir: Path) -> None:
    stale = (
        campaign_dir / "load-results" / "gemini" / MODEL / RUN_ID
        / "bracket-200rps-shards" / "shard-2.json"
    )
    stale.write_text('{"samples": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="mixed campaign"):
        build_summary(campaign_dir, RUN_ID, MODEL)
