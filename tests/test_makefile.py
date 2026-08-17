from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).parents[1]

# Every test in this module shells out to make; skip them all (rather than
# error) on hosts without it so the suite stays fully offline-safe.
pytestmark = pytest.mark.skipif(
    shutil.which("make") is None, reason="make is unavailable"
)


@pytest.mark.parametrize(
    ("provider", "model"),
    (
        ("gemini", "gemini-test"),
        ("together", "organization/model"),
    ),
)
@pytest.mark.parametrize(
    "target",
    ("provider-smoke", "quality", "capacity-ramp"),
)
def test_provider_make_targets_render_offline(
    provider: str,
    model: str,
    target: str,
) -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            target,
            f"PROVIDER={provider}",
            f"MODEL={model}",
            "RUN_ID=dry-run",
            "RETRY_RPS=1",
            "SOAK_RPS=1",
        ],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f'--provider "{provider}"' in result.stdout
    assert f'--model "{model}"' in result.stdout
    results_root = "eval-results" if target == "quality" else "load-results"
    model_path = model.replace("/", "-")
    expected_directory = PROJECT_DIR / results_root / provider / model_path / "dry-run"
    assert str(expected_directory) in result.stdout
    assert "gcloud" not in result.stdout


def test_synthetic_smoke_uses_its_own_grouped_result_directory() -> None:
    result = subprocess.run(
        ["make", "-n", "synthetic-smoke", "RUN_ID=dry-run"],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    expected_directory = PROJECT_DIR / "load-results" / "synthetic" / "local" / "dry-run"
    assert str(expected_directory) in result.stdout
    assert "gcloud" not in result.stdout


def test_capacity_ramp_renders_abort_thresholds() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "capacity-ramp",
            "PROVIDER=gemini",
            "MODEL=gemini-test",
            "RUN_ID=dry-run",
        ],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '--max-failure-rate "0.01"' in result.stdout
    assert '--max-service-p95-ms "5000"' in result.stdout


def test_capacity_ramp_enforces_the_total_request_budget() -> None:
    # With a 40-request budget and 30-second stages, the 1 RPS stage uses 30,
    # the 2 RPS stage is truncated to the remaining 10, and the 3 RPS stage
    # never runs. Budget exhaustion is a clean stop, not an error.
    true_path = shutil.which("true")
    assert true_path is not None

    result = subprocess.run(
        [
            "make",
            "capacity-ramp",
            f"PYTHON={true_path}",
            "PROVIDER=gemini",
            "RUN_ID=budget-dry-run",
            "RAMP_RPS=1 2 3",
            "RAMP_DURATION_SECONDS=30",
            "RAMP_REQUEST_BUDGET=40",
        ],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Running 30 requests against gemini at 1 RPS" in result.stdout
    assert "Running 10 requests against gemini at 2 RPS" in result.stdout
    assert "at 3 RPS" not in result.stdout
    assert "RAMP_REQUEST_BUDGET exhausted" in combined


def test_quality_factorial_runs_every_cell_and_repeat() -> None:
    # 2 thinking levels x 2 output caps x N repeats, each an independent
    # quality_eval invocation with its own artifact.
    true_path = shutil.which("true")
    assert true_path is not None

    result = subprocess.run(
        [
            "make",
            "quality-factorial",
            f"PYTHON={true_path}",
            "PROVIDER=gemini",
            "RUN_ID=factorial-dry-run",
            "FACTORIAL_REPEATS=2",
        ],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("Quality cell") == 8
    assert "thinking=default cap=none repeat=1" in result.stdout
    assert "thinking=default cap=256 repeat=2" in result.stdout
    assert "thinking=0 cap=none repeat=2" in result.stdout
    assert "thinking=0 cap=256 repeat=1" in result.stdout


def test_pending_request_limit_is_rendered_when_requested() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "capacity-ramp",
            "PROVIDER=gemini",
            "MODEL=gemini-test",
            "MAX_PENDING_REQUESTS=256",
            "RUN_ID=dry-run",
        ],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '--max-pending-requests "256"' in result.stdout
