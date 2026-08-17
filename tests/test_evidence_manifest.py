from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evidence_manifest import build_manifest

RUN_ID = "20260817T000000Z"


def provider_metadata(**overrides: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": "Gemini",
        "platform": "Vertex AI",
        "project": "secret-project-id",
        "location": "global",
        "model": "gemini-2.5-flash",
        "max_retries": 0,
        "max_output_tokens": None,
        "thinking_budget": None,
        "dependencies": {"google-genai": "2.13.0", "httpx": "0.28.1"},
    }
    metadata.update(overrides)
    return metadata


def load_artifact(
    *, rps: float, requests: int, failures: int = 0, retries: int = 0
) -> dict[str, Any]:
    return {
        "provider": provider_metadata(),
        "workload": {"sha256": "workload-digest", "question_count": 5},
        "config": {"requests_per_second": rps, "requests": requests},
        "runtime": {"python": "3.12.8", "platform": "test"},
        "requests": requests,
        "successes": requests - failures,
        "failures": failures,
        "failure_modes": {"http_429": failures} if failures else {},
        "successful_requests_per_second": rps * 0.98,
        "service_latency_ms": {"p50": 600.0, "p95": 1100.0, "p99": 1500.0},
        "queue_delay_ms": {"p95": 5.0},
        "warmup": {"requests": 5, "failures": 0},
        "retries": retries,
        "observed_input_tokens": requests * 40,
        "observed_output_tokens": requests * 20,
        "observed_thought_tokens": 0,
    }


def quality_artifact(
    *, thinking: int | None, cap: int | None, thought_tokens: int
) -> dict[str, Any]:
    return {
        "provider": provider_metadata(
            thinking_budget=thinking, max_output_tokens=cap
        ),
        "dataset": {"sha256": "dataset-digest", "version": 2},
        "cases": 10,
        "passed": 10,
        "failed": 0,
        "elapsed_seconds": 4.2,
        "observed_input_tokens": 433,
        "observed_output_tokens": 500,
        "observed_thought_tokens": thought_tokens,
    }


def write_artifacts(root: Path) -> None:
    load_dir = root / "load-results" / "gemini" / "gemini-2.5-flash" / RUN_ID
    synthetic_dir = root / "load-results" / "synthetic" / "local" / RUN_ID
    eval_dir = root / "eval-results" / "gemini" / "gemini-2.5-flash" / RUN_ID
    for directory in (load_dir, synthetic_dir, eval_dir):
        directory.mkdir(parents=True)

    files: dict[Path, dict[str, Any]] = {
        synthetic_dir / "00-synthetic-smoke.json": load_artifact(
            rps=0, requests=10000
        ),
        load_dir / "01-gemini-smoke.json": load_artifact(rps=1, requests=1),
        load_dir / "04-gemini-capacity-1rps.json": load_artifact(
            rps=1, requests=30
        ),
        load_dir / "04-gemini-capacity-10rps.json": load_artifact(
            rps=10, requests=300, failures=6
        ),
        load_dir / "05-gemini-retry-off-8rps.json": load_artifact(
            rps=8, requests=240
        ),
        load_dir / "06-gemini-retry-on-8rps.json": load_artifact(
            rps=8, requests=240, retries=3
        ),
        eval_dir
        / "02-gemini-quality-think-default-cap-none-r1.json": quality_artifact(
            thinking=None, cap=None, thought_tokens=400
        ),
        eval_dir
        / "02-gemini-quality-think-0-cap-256-r2.json": quality_artifact(
            thinking=0, cap=256, thought_tokens=0
        ),
    }
    for path, document in files.items():
        path.write_text(json.dumps(document), encoding="utf-8")


def test_build_manifest_aggregates_redacts_and_checksums(tmp_path: Path) -> None:
    write_artifacts(tmp_path)

    manifest = build_manifest(
        project_dir=tmp_path,
        run_id=RUN_ID,
        provider_dir="gemini",
        model_dir="gemini-2.5-flash",
    )

    assert manifest["schema_version"] == 2
    assert manifest["run_id"] == RUN_ID
    assert manifest["scope"] == {
        "provider": "Gemini",
        "platform": "Vertex AI",
        "location": "global",
        "model": "gemini-2.5-flash",
        "soak_performed": False,
        "raw_artifacts_gitignored": True,
    }
    assert manifest["runtime"]["python"] == "3.12.8"
    assert manifest["runtime"]["dependencies"]["google-genai"] == "2.13.0"

    # Live measured requests exclude the synthetic run: 1 + 30 + 300 + 240 + 240.
    assert manifest["load"]["measured_live_requests"] == 811
    assert manifest["load"]["measured_failures"] == 6
    assert manifest["load"]["measured_retries"] == 3
    assert manifest["load"]["warmup_requests"] == 25
    assert manifest["load"]["workload_sha256"] == "workload-digest"

    stages = manifest["load"]["capacity_stages"]
    assert [stage["offered_rps"] for stage in stages] == [1, 10]
    assert stages[1]["failures"] == 6
    assert stages[1]["failure_modes"] == {"http_429": 6}

    comparison = manifest["load"]["retry_comparison"]
    assert comparison["offered_rps"] == 8
    assert comparison["retries_off"]["max_retries"] == 0
    assert comparison["retries_on"]["measured_retries"] == 3

    cells = manifest["quality"]["cells"]
    assert manifest["quality"]["dataset_sha256"] == "dataset-digest"
    assert len(cells) == 2
    by_repeat = {cell["repeat"]: cell for cell in cells}
    assert by_repeat[1]["thinking_budget"] is None
    assert by_repeat[1]["observed_thought_tokens"] == 400
    assert by_repeat[2]["thinking_budget"] == 0
    assert by_repeat[2]["max_output_tokens"] == 256

    # Checksums cover every raw artifact, keyed by project-relative path.
    checksums = manifest["raw_artifact_sha256"]
    assert len(checksums) == 8
    smoke_path = (
        f"load-results/gemini/gemini-2.5-flash/{RUN_ID}/01-gemini-smoke.json"
    )
    expected_digest = hashlib.sha256(
        (tmp_path / smoke_path).read_bytes()
    ).hexdigest()
    assert checksums[smoke_path] == expected_digest

    # The project identifier must not appear anywhere in the manifest.
    assert "secret-project-id" not in json.dumps(manifest)
