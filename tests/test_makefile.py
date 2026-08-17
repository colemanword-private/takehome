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
    ("provider-smoke", "quality", "capacity-ramp", "retry-off", "soak"),
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


def test_thinking_control_is_rendered_only_when_requested() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "quality-controlled",
            "PROVIDER=gemini",
            "MODEL=gemini-test",
            "THINKING_BUDGET=0",
            "RUN_ID=dry-run",
        ],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '--thinking-budget "0"' in result.stdout


def test_model_default_is_owned_by_the_provider_config() -> None:
    # The Makefile must not branch on provider names to duplicate a default the
    # provider's configuration already owns; without MODEL it passes no --model
    # and groups results under the uniform "default" directory.
    result = subprocess.run(
        ["make", "-n", "provider-smoke", "PROVIDER=gemini", "RUN_ID=dry-run"],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--model" not in result.stdout
    expected_directory = PROJECT_DIR / "load-results" / "gemini" / "default" / "dry-run"
    assert str(expected_directory) in result.stdout


def test_quality_runs_controlled_eval_even_when_baseline_fails() -> None:
    # A failing baseline must not abort the paired experiment: the controlled
    # run still executes, and the combined exit status stays nonzero.
    result = subprocess.run(
        [
            "make",
            "quality",
            "PYTHON=false",
            "PROVIDER=gemini",
            "MODEL=gemini-test",
            "RUN_ID=dry-run",
        ],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert combined.count("Python environment is missing") == 2, combined


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
