"""Build the sanitized, checksummed evidence manifest for one campaign run.

Raw load and eval artifacts stay gitignored because they may contain model
outputs. This script aggregates them into a redacted summary — no prompts,
generated outputs, error messages, credentials, or project identifier — plus
a SHA-256 map so the raw artifacts can be verified if supplied separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPEAT_PATTERN = re.compile(r"-r(\d+)\.json$")
# Provider metadata fields that identify the deployment rather than the account.
_SCOPE_FIELDS = ("provider", "platform", "location", "model")


def build_manifest(
    *,
    project_dir: Path,
    run_id: str,
    provider_dir: str,
    model_dir: str,
) -> dict[str, Any]:
    load_dir = project_dir / "load-results" / provider_dir / model_dir / run_id
    synthetic_dir = project_dir / "load-results" / "synthetic" / "local" / run_id
    eval_dir = project_dir / "eval-results" / provider_dir / model_dir / run_id

    live_artifacts = _read_documents(load_dir)
    synthetic_artifacts = _read_documents(synthetic_dir)
    quality_artifacts = _read_documents(eval_dir)
    if not live_artifacts:
        raise ValueError(f"no live load artifacts found under {load_dir}")

    reference = next(iter(live_artifacts.values()))
    provider = reference["provider"]

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "scope": {
            **{field: provider.get(field) for field in _SCOPE_FIELDS},
            "soak_performed": False,
            "raw_artifacts_gitignored": True,
        },
        "runtime": {
            "python": reference["runtime"]["python"],
            "dependencies": provider.get("dependencies", {}),
        },
        "quality": _quality_section(quality_artifacts),
        "load": _load_section(live_artifacts),
        "raw_artifact_sha256": _checksums(
            project_dir,
            [*synthetic_artifacts, *live_artifacts, *quality_artifacts],
        ),
        "redaction": {
            "excluded": [
                "project identifier",
                "credentials",
                "prompts",
                "generated outputs",
                "error messages",
            ],
            "note": (
                "The local raw artifacts can be verified against the SHA-256 "
                "manifest and supplied separately if needed."
            ),
        },
    }
    return manifest


def _read_documents(directory: Path) -> dict[Path, dict[str, Any]]:
    if not directory.is_dir():
        return {}
    return {
        path: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


def _quality_section(
    artifacts: dict[Path, dict[str, Any]],
) -> dict[str, Any] | None:
    if not artifacts:
        return None
    cells = []
    reference_dataset = next(iter(artifacts.values()))["dataset"]
    for path, report in artifacts.items():
        provider = report["provider"]
        repeat_match = _REPEAT_PATTERN.search(path.name)
        cells.append(
            {
                "artifact": path.name,
                "repeat": int(repeat_match.group(1)) if repeat_match else None,
                "thinking_budget": provider.get("thinking_budget"),
                "max_output_tokens": provider.get("max_output_tokens"),
                "max_retries": provider.get("max_retries"),
                "cases": report["cases"],
                "passed": report["passed"],
                "failed": report["failed"],
                "elapsed_seconds": report["elapsed_seconds"],
                "observed_input_tokens": report["observed_input_tokens"],
                "observed_output_tokens": report["observed_output_tokens"],
                "observed_thought_tokens": report.get(
                    "observed_thought_tokens", 0
                ),
            }
        )
    return {
        "dataset_sha256": reference_dataset["sha256"],
        "dataset_version": reference_dataset["version"],
        "cells": cells,
    }


def _load_section(artifacts: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    reference = next(iter(artifacts.values()))
    section: dict[str, Any] = {
        "workload_sha256": reference["workload"]["sha256"],
        "measured_live_requests": sum(
            report["requests"] for report in artifacts.values()
        ),
        "measured_failures": sum(
            report["failures"] for report in artifacts.values()
        ),
        "measured_retries": sum(
            report["retries"] or 0 for report in artifacts.values()
        ),
        "warmup_requests": sum(
            report["warmup"]["requests"] for report in artifacts.values()
        ),
        "capacity_stages": [
            _stage_summary(report)
            for _, report in sorted(
                (
                    (report["config"]["requests_per_second"], report)
                    for path, report in artifacts.items()
                    if "-capacity-" in path.name
                ),
                key=lambda pair: pair[0],
            )
        ],
    }

    retries_off = _single_artifact(artifacts, "-retry-off-")
    retries_on = _single_artifact(artifacts, "-retry-on-")
    if retries_off and retries_on:
        section["retry_comparison"] = {
            "offered_rps": retries_off["config"]["requests_per_second"],
            "requests_per_run": retries_off["requests"],
            "retries_off": _retry_summary(retries_off),
            "retries_on": _retry_summary(retries_on),
        }
    return section


def _stage_summary(report: dict[str, Any]) -> dict[str, Any]:
    latency = report["service_latency_ms"]
    return {
        "offered_rps": report["config"]["requests_per_second"],
        "requests": report["requests"],
        "failures": report["failures"],
        "failure_modes": report["failure_modes"],
        "successful_rps": report["successful_requests_per_second"],
        "service_latency_ms": {
            "p50": latency["p50"],
            "p95": latency["p95"],
            "p99": latency["p99"],
        },
        "queue_delay_p95_ms": report["queue_delay_ms"]["p95"],
    }


def _retry_summary(report: dict[str, Any]) -> dict[str, Any]:
    stage = _stage_summary(report)
    return {
        "max_retries": report["provider"].get("max_retries"),
        "successful_rps": stage["successful_rps"],
        "measured_failures": report["failures"],
        "measured_retries": report["retries"] or 0,
        "service_latency_ms": stage["service_latency_ms"],
        "queue_delay_p95_ms": stage["queue_delay_p95_ms"],
    }


def _single_artifact(
    artifacts: dict[Path, dict[str, Any]], marker: str
) -> dict[str, Any] | None:
    matches = [
        report for path, report in artifacts.items() if marker in path.name
    ]
    if len(matches) > 1:
        raise ValueError(f"expected at most one artifact matching {marker!r}")
    return matches[0] if matches else None


def _checksums(project_dir: Path, paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(project_dir).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(paths)
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate one run's artifacts into a sanitized manifest."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--provider-dir", default="gemini")
    parser.add_argument("--model-dir", default="gemini-2.5-flash")
    parser.add_argument(
        "--project-dir", type=Path, default=Path(__file__).parent
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = build_manifest(
            project_dir=args.project_dir,
            run_id=args.run_id,
            provider_dir=args.provider_dir,
            model_dir=args.model_dir,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    output = args.output or (
        args.project_dir
        / "evidence"
        / args.provider_dir
        / args.model_dir
        / f"{args.run_id}-summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{rendered}\n", encoding="utf-8")
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
