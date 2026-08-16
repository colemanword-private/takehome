from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).parents[1]


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
    if shutil.which("make") is None:
        pytest.skip("make is unavailable")

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
    assert "dry-run" in result.stdout
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
